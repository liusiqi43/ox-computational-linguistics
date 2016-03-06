import numpy as np

from parser import START, counter, parse, trigramize, id_to_token, build_dict
from glob import glob

####
# seq: ['**start**', 'we', 'are', 'drinking', 'milktea', '**end**']
####
def viterbi(seq, transition, emission, word2id, tag2id):
    assert seq[0] == START, seq
    seq = [word2id[part[0]] if part[0] in word2id else len(word2id) for part in seq]
    seq = seq[1:]

    scores = np.zeros((len(transition), len(seq)))
    backpointer = np.zeros((len(transition), len(seq)), dtype=int)

    assert tag2id[START[1]] == 0
    scores[:, 0] = np.log(transition[tag2id[START[1]], :]) + np.log(emission[:, seq[0]])
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
    tag2id, word2id = build_dict(parsed[:-10])
    id2word = {v:k for k, v in word2id.iteritems()}
    id2tag = {v:k for k, v in tag2id.iteritems()}
    emission, transition = counter(parsed[:-10], tag2id, word2id)
    print 'test POS', parsed[-1][1:-1]
    output = viterbi(parsed[-1], transition, emission, word2id, tag2id)
    print 'TAGGED', id_to_token(output, id2word, id2tag)
