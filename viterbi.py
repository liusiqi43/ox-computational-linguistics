import numpy as np

from parser import EPS, START, END, counter, parse
from glob import glob

def _emit_prob(emission, cat, word):
    if (word, cat) == START or (word, cat) == END:
        return 1
    if cat == START[1] or cat == END[1]:
        return EPS
    return emission[cat][word] if word in emission[cat] else 1

####
# seq: ['We', 'are', 'drinking', 'milktea', '**end**']
####
def viterbi(seq, transition, emission):
    assert seq[0] == START[0] and seq[-1] == END[0], seq
    seq = seq[1:]

    # Fix categories order.
    categories = transition.keys()

    scores = np.zeros((len(categories), len(seq)))
    backpointer = np.zeros((len(categories), len(seq)), dtype=int)
    seq = [s.lower() for s in seq]

    for i, cat in enumerate(categories):
        scores[i, 0] = np.log(transition[START[1]][cat]) \
                        + np.log(_emit_prob(emission, cat, seq[0]))

    for j in xrange(1, len(seq)):
        for i, cat in enumerate(categories):
            max_k, max_score = -1, -np.inf
            for k, c in enumerate(categories):
                k_score = scores[k, j-1] + np.log(transition[c][cat]) \
                            + np.log(_emit_prob(emission, cat, seq[j]))
                if k_score > max_score:
                    max_k, max_score = k, k_score
            scores[i, j] = max_score
            backpointer[i, j] = max_k

    j = int(np.argmax(scores, axis=0)[-1])
    sol = [categories[j]]
    for i in xrange(len(seq)-1, 0, -1):
        j = backpointer[j, i]
        sol.append(categories[j])
    sol.reverse()
    return zip(seq[:-1], sol[:-1])


if __name__ == '__main__':
    path = '../WSJ-2-12/*/*.POS'
    docs = glob(path)

    parsed = parse(docs)

    np.random.shuffle(parsed)
    emission, transition = counter(parsed[:-1])

    test_seq = [part[0].lower() for part in parsed[0]]
    print 'test POS', parsed[0][1:-1]
    output = viterbi(test_seq, transition, emission)
    print 'TAGGED', output
