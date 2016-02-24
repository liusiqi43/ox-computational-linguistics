from parser import parse, counter, id2tag, trigramize
from viterbi import viterbi
from collections import defaultdict

import numpy as np

DEBUG = False

def _strip_pos(seq):
    return [part[0] for part in seq]

def _fold(parsed, k):
    fold_size = len(parsed)/k
    start = 0
    while start + fold_size < len(parsed):
        # yield train, test.
        yield parsed[:start] + parsed[start+fold_size:], parsed[start:start+fold_size]
        start += fold_size

def _compare(target, output):
    assert len(target) == len(output)
    count_ok = 0
    for i in xrange(len(target)):
        assert(target[i][0] == output[i][0]), 'input words changed.' + str(target) + str(output)
        # keep only the last (current) tag. Consider DET-NNP, VBN-NNP correct labelling.
        output_tag = id2tag[output[i][1]].split('-')[-1]
        target_tags = id2tag[target[i][1]].split('-')[-1].split('|')
        if output_tag in target_tags:
            count_ok += 1
        elif DEBUG:
            print target, '\n!=\n', output
            print target[i], '\n!=\n', output[i], '\n'
    return count_ok, len(target)

def _counter_known(parsed, train, known, discount):
    emission, transition = None, None
    if known:
        emission, _ = counter(parsed, discount)
        _, transition = counter(train, discount)
    else:
        emission, transition = counter(train, discount)

    return emission, transition

def k_fold_cross_valid_known(k, parsed, known, discounts):
    res = defaultdict(list)
    for train, test in _fold(parsed, k):
        for discount in discounts:
            print 'train: ', len(train), 'test: ', len(test)
            emission, transition = _counter_known(parsed, train, known, discount)

            count_ok, count_total = 0., 0.
            for i, seq in enumerate(test):
                stripped_seq = _strip_pos(seq)
                out = viterbi(stripped_seq, transition, emission)
                ok, total = _compare(seq[1:-1], out)
                count_ok += ok; count_total += total
                print 'evaluating', i, 'th sentence.', count_ok/count_total, 'so far.'
            res[discount].append(count_ok/count_total)
            print 'Fold accuracy: ', res[discount][-1], 'discount: ', discount
    for d in res:
        print 'discount:', d, '->', 'avg:', np.mean(res[d])


if __name__ == '__main__':
    from glob import glob

    path = '../WSJ-2-12/*/*.POS'
    docs = glob(path)

    parsed = parse(docs)
    parsed = trigramize(parsed)

    np.random.seed(647)
    np.random.shuffle(parsed)
    k = 10
    # Specify all discount factors to try.
    discounts = [0.85]
    print k, 'fold validation unknown:'
    k_fold_cross_valid_known(k, parsed, False, discounts)
    print k, 'fold validation known:'
    k_fold_cross_valid_known(k, parsed, True, discounts)
