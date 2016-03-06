from parser import parse, counter, trigramize, id_to_token, build_dict
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
        output_tag = output[i][1].split('-')[-1]
        target_tags = target[i][1].split('-')[-1].split('|')
        if output_tag in target_tags:
            count_ok += 1
    return count_ok, len(target)

def _counter_known(parsed, train, known, discount, tag2id, word2id, prior):
    emission, transition = None, None
    if known:
        emission, _ = counter(parsed, tag2id, word2id, discount, prior)
        _, transition = counter(train, tag2id, word2id, discount, prior)
    else:
        emission, transition = counter(train, tag2id, word2id, discount, prior)

    return emission, transition

def k_fold_cross_valid_known(k, parsed, known, discounts):
    res = defaultdict(list)
    for train, test in _fold(parsed, k):
        for discount in discounts:
            print 'train: ', len(train), 'test: ', len(test)
            tag2id, word2id = build_dict(parsed)
            id2tag = {v: k for k, v in tag2id.iteritems()}
            id2word = {v: k for k, v in word2id.iteritems()}
            emission, transition = _counter_known(parsed, train, known,
                                                  0.85, tag2id, word2id, discount)

            count_ok, count_total = 0., 0.
            for i, seq in enumerate(test):
                out = viterbi(seq, transition, emission, word2id, tag2id)
                ok, total = _compare(seq[1:-1], id_to_token(out, id2word, id2tag))
                count_ok += ok; count_total += total
                if DEBUG:
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
    parsed = parsed
    k = 10
    # Specify all discount factors to try.
    discounts = [0.3]
    print k, 'fold validation unknown:'
    k_fold_cross_valid_known(k, parsed, False, discounts)
    print k, 'fold validation known:'
    k_fold_cross_valid_known(k, parsed, True, discounts)
