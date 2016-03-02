import numpy as np

from parser import EPS, START, END, counter, parse, trigramize, word2id, translate_seq, id2tag
from glob import glob

####
# seq: ['**start**', 'we', 'are', 'drinking', 'milktea', '**end**']
####
def viterbi(seq, transition, emission):
    assert seq[0] == START[0] and seq[-1] == END[0], seq
    seq = seq[1:]

    scores = np.zeros((len(transition), len(seq)))
    backpointer = np.zeros((len(transition), len(seq)), dtype=int)

    # If word is unknown in emission, assign it to the unknown element, which has emission EPS.
    seq[0] = min(seq[0], emission.shape[1])
    scores[:, 0] = np.log(transition[START[1]][:]) + np.log(emission[:, seq[0]])
    for j in xrange(1, len(seq)):
        for i in xrange(len(transition)):
            k_score = scores[:, j-1] + np.log(transition[:, i]) + np.log(emission[i, seq[j]])
            backpointer[i, j] = np.argmax(k_score)
            scores[i, j] = k_score[backpointer[i, j]]

    j = int(np.argmax(scores, axis=0)[-1])
    sol = [j]
    for i in xrange(len(seq)-1, 0, -1):
        j = backpointer[j, i]
        sol.append(j)
    sol.reverse()
    return zip(seq[:-1], sol[:-1])


if __name__ == '__main__':
    path = '../WSJ-2-12/*/*.POS'
    docs = glob(path)

    parsed = parse(docs)

    np.random.shuffle(parsed)
    parsed = trigramize(parsed)
    emission, transition = counter(parsed[:-10])

    test_seq = [part[0] for part in parsed[-1]]
    print 'test POS', translate_seq(parsed[-1][1:-1])
    output = viterbi(test_seq, transition, emission)
    print 'TAGGED', translate_seq(output)
