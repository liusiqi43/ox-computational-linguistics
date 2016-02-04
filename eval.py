from parser import parse, counter
from viterbi import viterbi

import numpy as np

def _strip_pos(seq):
    return [part[0] for part in seq]

def _fold(parsed, k):
    fold_size = len(parsed)/k
    start = 0
    while start + fold_size < len(parsed):
        # yield train, test.
        yield parsed[:start] + parsed[start+fold_size:], parsed[start:start+fold_size]
        start += fold_size

def _compare(s1, s2):
    assert len(s1) == len(s2)
    count_ok = 0
    for i in xrange(len(s1)):
        if s1[i] == s2[i]:
            count_ok += 1
    return count_ok, len(s1)

def _counter_known(parsed, train, known):
    emission, transition = None, None
    if known:
        emission, _ = counter(parsed)
        _, transition = counter(train)
    else:
        emission, transition = counter(train)

    return emission, transition

def k_fold_cross_valid_known(k, parsed, known):
    res = []
    for train, test in _fold(parsed, k):
        print 'train: ', len(train), 'test: ', len(test)
        emission, transition = _counter_known(parsed, train, known)

        count_ok, count_total = 0., 0.
        for seq in test:
            stripped_seq = _strip_pos(seq)
            out = viterbi(stripped_seq, transition, emission)
            ok, total = _compare(seq[1:-1], out)
            count_ok += ok; count_total += total
        res.append(count_ok/count_total)
        print 'Fold accuracy: ', res[-1]
    print 'Avg: ', np.mean(res)


if __name__ == '__main__':
    from glob import glob

    path = '../WSJ-2-12/*/*.POS'
    docs = glob(path)

    parsed = parse(docs)

    np.random.seed(647)
    np.random.shuffle(parsed)
    k = 10
    print k, 'fold validation known:'
    k_fold_cross_valid_known(k, parsed, True)
    print k, 'fold validation unknown:'
    k_fold_cross_valid_known(k, parsed, False)
