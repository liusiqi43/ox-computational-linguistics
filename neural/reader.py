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
        self.seed = 647
        self.atomize = atomize
        self.split = split

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

    def _build_vocab(self, merged):
        words, tags = map(set, zip(*merged))
        self.word_to_id = dict(zip(words, range(len(words))))
        self.tag_to_id = dict(zip(tags, range(len(tags))))

    def _to_ids(self, merged):
        res = []
        for item in merged:
            res.append((self.word_to_id[item[0]], self.tag_to_id[item[1]]))
        return res

    def _get_datasets(self, merged, split):
        merged = self._to_ids(merged)
        train_size = int(len(merged) * split)
        train = merged[:train_size]
        test = merged[train_size:]
        x_train, y_train = zip(*train)
        x_test, y_test = zip(*test)
        return map(np.asarray, [x_train, y_train, x_test, y_test])

    def get_data(self, docs):
        parsed = self._raw_parse(docs)
        if self.atomize:
            res = []
            for seq in parsed:
                res += _atomize(seq)
            parsed = res
        np.random.seed(self.seed)
        np.random.shuffle(parsed)
        merged = _merge(parsed)
        self._build_vocab(merged)
        return self._get_datasets(merged, self.split)

    @staticmethod
    def iterator(x, y, batch_size, num_steps):
        """Iterate on the WSJ data.
        """
        data_len = len(x)
        batch_len = data_len // batch_size
        data_x = np.zeros([batch_size, batch_len], dtype=np.int32)
        data_y = np.zeros([batch_size, batch_len], dtype=np.int32)
        for i in range(batch_size):
            data_x[i] = x[batch_len * i:batch_len * (i + 1)]
            data_y[i] = y[batch_len * i:batch_len * (i + 1)]
            epoch_size = (batch_len - 1) // num_steps
            if epoch_size == 0:
                raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

        for i in range(epoch_size):
            x_ = data_x[:, i*num_steps:(i+1)*num_steps]
            y_ = data_y[:, i*num_steps:(i+1)*num_steps]
            yield (x_, y_)
