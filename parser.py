from __future__ import division

from glob import glob
from collections import defaultdict
from string import maketrans
from copy import deepcopy

import operator
import copy

START = ('**start**', 'START')
END = ('**end**', 'END')
EPS = 1e-32

word2id = dict()
id2word = dict()
tag2id = dict()
id2tag = dict()

word2id[START[0]] = len(word2id)
word2id[END[0]] = len(word2id)
tag2id[START[1]] = len(tag2id)
tag2id[END[1]] = len(tag2id)

START = (word2id[START[0]], tag2id[START[1]])
END = (word2id[END[0]], tag2id[END[1]])

id2word[START[0]] = '**start**'
id2word[END[0]] = '**end**'
id2tag[START[1]] = 'START'
id2tag[END[1]] = 'END'

def _fill_dicts(word, tag):
    if word is not None and word not in word2id:
        word2id[word] = len(word2id)
        id2word[word2id[word]] = word
    if tag is not None and tag not in tag2id:
        tag2id[tag] = len(tag2id)
        id2tag[tag2id[tag]] = tag
    assert len(word2id) == len(id2word)
    assert len(tag2id) == len(id2tag)
    return word2id[word] if word is not None else None, tag2id[tag] if tag is not None else None

def _is_num(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

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
            _fill_dicts(None, t)
            res.append(t)
    return res


def _normalize(counts, discount = 0):
    for key1 in counts:
        total = 0.
        for key2 in counts[key1]:
            total += counts[key1][key2]

        for key2 in counts[key1]:
            counts[key1][key2] = max(counts[key1][key2] - discount, 0) / total
    return


def parse(docs):
    parsed = []
    for doc in docs:
        print doc
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
                    id_word, id_tag = _fill_dicts(p[0], p[1])
                    seq.append((id_word, id_tag))

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

# returns (new) id for t1-t2 tag.
def _combine_tag(t1, t2):
    _, t = _fill_dicts(None, id2tag[t1]+'-'+id2tag[t2])
    return t

def trigramize(parsed):
    trigram_parsed = []
    for seq in parsed:
        trigram_seq = []
        # item = (word, tag)
        for i, item in enumerate(seq):
            # Don't change anything if it's the start tag.
            if i == 0:
                trigram_seq.append(item)
                continue
            # trigram_item = (word, prevtag-tag)
            trigram_item = (item[0], _combine_tag(seq[i-1][1], item[1]))
            trigram_seq.append(trigram_item)
        trigram_parsed.append(trigram_seq)
    return trigram_parsed

def _kneser_ney_smoothing(bigram, unigram, discount):
    pairs_count = 0

    for fromtag in bigram:
        for totag in bigram[fromtag]:
            if bigram[fromtag][totag] > 1:
                pairs_count += 1

    _normalize(bigram, discount)

    for fromtag in unigram:
        for totag in unigram:
            _lambda = (discount / unigram[fromtag]) * len(filter(lambda t: bigram[fromtag][t] > 1, bigram[fromtag]))
            discounted = bigram[fromtag][totag]
            bigram[fromtag][totag] = max(EPS, discounted + _lambda * sum([(bigram[t][totag] > 1) for t in bigram]) / pairs_count)
    return bigram

def counter(parsed, discount = 0.75):
    emission = defaultdict(dict)
    transition = defaultdict(dict)

    categories = {START[1] : 1, END[1] : 1}
    vocabulary = set((START[0], END[0]))

    for seq in parsed:
        for part in seq:
            vocabulary.add(part[0])
            for tag in _split_tags(id2tag[part[1]]):
                if tag not in categories:
                    categories[tag2id[tag]] = 0
                categories[tag2id[tag]] += 1

    # Add one to avoid zeros.
    for c1 in categories:
        for c2 in categories:
            transition[c1][c2] = 1.

    # No tag transit from END or to START.
    for c in categories:
        transition[END[1]][c] = EPS
        transition[c][START[1]] = EPS

    for seq in parsed:
        for i in xrange(len(seq)):
            # record emission count for ith part.
            tags = _split_tags(id2tag[seq[i][1]])
            for tag in tags:
                s = seq[i][0]
                if s not in emission[tag2id[tag]]:
                    emission[tag2id[tag]][s] = 0
                emission[tag2id[tag]][s] += 1

            if i == 0:
                continue

            tags_prev = _split_tags(id2tag[seq[i-1][1]])
            # trainsition count from t1 to t2.
            for t1 in tags_prev:
                for t2 in tags:
                    transition[tag2id[t1]][tag2id[t2]] += 1

    _normalize(emission)
    # transition: bigram, categories: unigram.
    _kneser_ney_smoothing(transition, categories, discount)


    return emission, transition

def translate_seq(seq):
    return [(id2word[item[0]], id2tag[item[1]]) for item in seq]


if __name__ == '__main__':
    path = '../WSJ-2-12/*/*.POS'
    docs = glob(path)
    parsed = parse(docs)
    parsed = trigramize(parsed)
    # emission, transition = counter(parsed)
    print translate_seq(parsed[10])
