from __future__ import division

from glob import glob
from collections import defaultdict
from string import maketrans
from copy import deepcopy

import operator
import numpy as np
import copy

START = ('**start**', 'START')
END = ('**end**', 'END')
EPS = 1e-32

def _split_tags(tag):
    res = []
    # t1|t2|t3-t4|t5|t6
    # to t1-t4, t1-t5, t1-t6, t2-t4, t2-t5, t2-t6, t3-t4, t3-t5, t3-t6
    tags = tag.split('-')
    assert(len(tags) <= 2), tag + ' has more than 2 parts'
    if len(tags) == 1:
        return tags[0].split('|')
    t1s = tags[0].split('|')
    t2s = tags[1].split('|')
    for t1 in t1s:
        for t2 in t2s:
            t = t1+'-'+t2
            res.append(t)
    return res


def _normalize(counts, discount = 0):
    total = np.sum(counts, 1)
    counts = np.maximum(counts - discount, 0) / total[:, None]
    return counts

def parse(docs):
    parsed = []
    for doc in docs:
        with open(doc, 'r') as f:
            seq = [START]
            for line in f:
                line = line.translate(None, '[]')
                # Empty line.
                if len(line) == 0:
                    continue
                # Stop sequence line.
                if line.strip() == len(line.strip()) * '=':
                    if len(seq) > 1:
                        seq.append(END); parsed.append(seq)
                        seq = [START]
                    continue
                parts = [item.strip().rsplit('/', 1) for item in line.split()]
                for p in parts:
                    if p[1] == 'CD':
                        p[0] = '__num__'
                    seq.append(p)

                    # End of sequence.
                    if p[0] in ['.', '?', '!']:
                        seq.append(END); parsed.append(seq)
                        seq = [START]
                        continue
        if len(seq) > 1:
            assert parsed[-1][-1] == END and seq[0] == START
            del parsed[-1][-1]; del seq[0]
            seq.append(END); parsed[-1].extend(seq)
            print 'extended', parsed[-1]
    return parsed

def trigramize(parsed):
    trigram_parsed = []
    for seq in parsed:
        trigram_seq = []
        # item = (word, tag)
        for i in xrange(len(seq)):
            # Don't change anything if it's the start tag.
            if i == 0:
                trigram_seq.append(seq[i])
                continue
            # trigram_item = (word, prevtag-tag)
            trigram_item = (seq[i][0], seq[i-1][1] + '-' + seq[i][1])
            trigram_seq.append(trigram_item)
        trigram_parsed.append(trigram_seq)
    return trigram_parsed

def _unfold_transition(transition, tag2id):
    if max(['-' in t for t in tag2id]) == False:
        return None, None
    uf_tag2id = {END[1] : 0}
    uf_transition = defaultdict(dict)
    for fromtag in tag2id:
        for totag in tag2id:
            if transition[tag2id[fromtag], tag2id[totag]] <= EPS:
                continue
            # Approximation here.
            l, r = totag.split('-', 1)[0], totag.rsplit('-', 1)[-1]
            if r not in uf_transition[l]:
                uf_transition[l][r] = 0.
            uf_transition[l][r] += transition[tag2id[fromtag], tag2id[totag]]
            _get_id(uf_tag2id, l)

    res = np.zeros([len(uf_tag2id), len(uf_tag2id)]) + EPS
    # Add one to all valid transitions.
    for fromtag in uf_transition:
        for totag in uf_transition[fromtag]:
            if _valid_transition(fromtag, totag):
                res[uf_tag2id[fromtag], uf_tag2id[totag]] = 1. + uf_transition[fromtag][totag]
    return res, uf_tag2id

def _kneser_ney_smoothing(transition, tag2id, discount):
    uf_transition, uf_tag2id = _unfold_transition(transition, tag2id)
    if uf_transition is None:
        return _normalize(transition)
    kn_transition = _kneser_ney_smoothing(uf_transition, uf_tag2id, discount)

    result = _normalize(transition, discount)
    for fromtag, fid in tag2id.iteritems():
        for totag, tid in tag2id.iteritems():
            l = uf_tag2id[totag.split('-', 1)[0]]
            r = uf_tag2id[totag.rsplit('-', 1)[-1]]
            gamma = (discount / np.sum(transition[fid, :])
                     * np.sum(transition[fid, :] >= 1))
            result[fid, tid] += gamma * kn_transition[l, r]
    return result

def _valid_transition(t1, t2):
    return t1.rsplit('-', 1)[-1] == t2.split('-', 1)[0]

def _get_id(d, w):
    if w not in d:
        d[w] = len(d)
    return d[w]

def build_dict(parsed):
    # build dictionaries.
    tag2id, word2id = {}, {}
    for seq in parsed:
        for (word, tag) in seq:
            for tag in _split_tags(tag):
                _get_id(tag2id, tag)
            _get_id(word2id, word)
    return tag2id, word2id


def counter(parsed, tag2id, word2id, discount, invalid_prior):
    print 'counting emission/transition matrices.'

    transition = np.zeros((len(tag2id), len(tag2id)))
    # Add one to all valid transitions.
    for fromtag in tag2id:
        for totag in tag2id:
            if _valid_transition(fromtag, totag):
                transition[tag2id[fromtag], tag2id[totag]] += 1.
            else:
                transition[tag2id[fromtag], tag2id[totag]] += invalid_prior

    # last column reserved for unknown words, with EPS as probability.
    emission = np.zeros((len(tag2id), len(word2id)+1)) + EPS
    for seq in parsed:
        for i in xrange(len(seq)):
            # record emission count for ith part.
            tags = _split_tags(seq[i][1])
            for tag in tags:
                word = seq[i][0]
                emission[tag2id[tag], word2id[word]] += 1

            if i == 0:
                continue

            tags_prev = _split_tags(seq[i-1][1])
            # trainsition count from t1 to t2.
            for t1 in tags_prev:
                for t2 in tags:
                    transition[tag2id[t1], tag2id[t2]] += 1

    emission = np.maximum(EPS, _normalize(emission))
    transition = np.maximum(EPS, _normalize(transition))
    # transition = np.maximum(EPS,
    #                         _kneser_ney_smoothing(transition, tag2id, discount))
    print 'done.'
    return emission, transition

def id_to_token(seq, id2word, id2tag):
    return [(id2word[item[0]] if item[0] in id2word else '**unk**',
             id2tag[item[1]]) for item in seq]

def token_to_id(seq, word2id, tag2id):
    return [(word2id[item[0]] if item[0] in word2id else len(word2id),
             tag2id[item[1]]) for item in seq]

if __name__ == '__main__':
    path = '../WSJ-2-12/*/*.POS'
    docs = glob(path)
    parsed = parse(docs)
    parsed = trigramize(parsed)
    tag2id, word2id = build_dict(parsed)
    id2word = {v:k for k, v in word2id.iteritems()}
    emission, transition = counter(parsed, tag2id, word2id)
    print parsed[10]
