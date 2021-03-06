from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np

from glob import glob
from collections import defaultdict

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

def _atomize(seq):
    res = []
    r = [[]]
    for item in seq:
        tags = _split_tags(item[1])
        r *= len(tags)
        for t in tags:
            for seq in r:
                seq.append((item[0], t))
    res += r
    return res



def _merge(parsed):
    merged = []
    for seq in parsed:
        merged += seq
    return merged

class Reader(object):
    def __init__(self, atomize=True, split=0.9):
        self.START = ('**start**', 'START')
        self.END = ('**end**', 'END')
        self.PAD = ('**pad**', 'PAD')
        self.seed = 42
        self.atomize = atomize
        self.split = split
        self.maxlen = -1
        self.ignore_ids = None

    def _pad(self, parsed):
        buckets = np.percentile([len(s) for s in parsed], range(0, 101, 10))
        self.maxlen = buckets[-2] # 90 percentile.
        print('pad all sentences to', self.maxlen)
        res = []
        for seq in parsed:
            if len(seq) > self.maxlen:
                continue
            res.append(seq + [self.PAD] * (self.maxlen - len(seq)))
        return res

    def _raw_parse(self, docs):
        parsed = []
        for doc in docs:
            with open(doc, 'r') as f:
                seq = [self.START]
                for line in f:
                    line = line.translate(None, '[]')
                    # Empty line.
                    if len(line) == 0:
                        continue
                    # Stop sequence line.
                    if line.strip() == len(line.strip()) * '=':
                        if len(seq) > 1:
                            seq.append(self.END); parsed.append(seq)
                            seq = [self.START]
                        continue
                    parts = [item.strip().rsplit('/', 1) for item in line.split()]
                    for p in parts:
                        if p[1] == 'CD':
                            p[0] = '**num**'
                        seq.append((p[0], p[1]))

                        # End of sequence.
                        if p[0] in ['.', '?', '!']:
                            seq.append(self.END); parsed.append(seq)
                            seq = [self.START]
                            continue
            if len(seq) > 1:
                assert parsed[-1][-1] == self.END and seq[0] == self.START
                del parsed[-1][-1]; del seq[0]
                seq.append(self.END); parsed[-1].extend(seq)
                print('extended', parsed[-1])
        return parsed

    def _build_vocab(self, padded):
        merged = _merge(padded)
        words, tags = map(set, zip(*merged))
        self.word_to_id = dict(zip(words, range(len(words))))
        self.tag_to_id = dict(zip(tags, range(len(tags))))
        self.ignore_ids = [self.tag_to_id[self.START[1]],
                           self.tag_to_id[self.END[1]],
                           self.tag_to_id[self.PAD[1]]]

    def _to_ids(self, padded):
        res = []
        for seq in padded:
            s = []
            for item in seq:
                s.append((self.word_to_id[item[0]], self.tag_to_id[item[1]]))
            res.append(s)
        return res

    def _split_xy(self, padded):
        x = np.zeros([len(padded), self.maxlen], dtype=np.int32)
        y = np.zeros([len(padded), self.maxlen], dtype=np.int32)
        mask = np.ones([len(padded), self.maxlen], dtype=np.bool)

        for i, seq in enumerate(padded):
            x[i], y[i] = map(np.asarray, zip(*seq))

        for ignored in self.ignore_ids:
            mask = np.logical_and(mask, y != ignored)
        return x, y, mask

    def _get_datasets(self, padded, split):
        padded = self._to_ids(padded)
        train_size = int(len(padded) * split)
        train = padded[:train_size]
        test = padded[train_size:]
        x_train, y_train, mask_train = self._split_xy(train)
        x_test, y_test, mask_test = self._split_xy(test)
        return x_train, y_train, mask_train, x_test, y_test, mask_test

    def get_data(self, docs):
        parsed = self._raw_parse(docs)
        if self.atomize:
            res = []
            for seq in parsed:
                res += _atomize(seq)
            parsed = res
        np.random.seed(self.seed)
        np.random.shuffle(parsed)
        parsed = self._pad(parsed)
        self._build_vocab(parsed)
        return self._get_datasets(parsed, self.split)

    @staticmethod
    def iterator(x, y, mask, batch_size):
        """Iterate on the WSJ data.
        """
        epoch_size = (len(x)-1) // batch_size
        for i in range(epoch_size):
            yield (x[i*batch_size:(i+1)*batch_size], y[i*batch_size:(i+1)*batch_size],
                   mask[i*batch_size:(i+1)*batch_size])
