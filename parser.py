from glob import glob
from collections import defaultdict
from string import maketrans

import operator

START = ('**start**', 'START')
END = ('**end**', 'END')
EPS = 1e-128

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
    if not word in word2id:
        word2id[word] = len(word2id)
        id2word[word2id[word]] = word
    if not tag in tag2id:
        tag2id[tag] = len(tag2id)
        id2tag[tag2id[tag]] = tag
    assert len(word2id) == len(id2word)
    assert len(tag2id) == len(id2tag)
    return word2id[word], tag2id[tag]

def _is_num(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def _atomize(categories):
    atoms = set([])
    for c in categories:
        for s in id2tag[c].split('|'):
            atoms.add(tag2id[s])
    return atoms

def _normalize(counts):
    for key1 in counts:
        total = 0.
        for key2 in counts[key1]:
            total += counts[key1][key2]

        for key2 in counts[key1]:
            counts[key1][key2] /= total
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

def counter(parsed):
    emission = defaultdict(dict)
    transition = defaultdict(dict)

    categories = set((START[1], END[1]))
    vocabulary = set((START[0], END[0]))

    for seq in parsed:
        for part in seq:
            categories.add(part[1])
            vocabulary.add(part[0])

    atoms = _atomize(categories)
    # trainsition count from c1 to c2 with smoothing.
    for c1 in atoms:
        for c2 in atoms:
            transition[c1][c2] = 1.

    # No tag transit from END or to START.
    for c in atoms:
        transition[END[1]][c] = EPS
        transition[c][START[1]] = EPS

    for seq in parsed:
        for i in xrange(len(seq)):
            # record emission count for ith part.
            tags = id2tag[seq[i][1]].split('|')
            for tag in tags:
                s = seq[i][0]
                if s not in emission[tag2id[tag]]:
                    emission[tag2id[tag]][s] = 0
                emission[tag2id[tag]][s] += 1

            if i == 0:
                continue

            tags_prev = id2tag[seq[i-1][1]].split('|')
            # trainsition count from t1 to t2.
            for t1 in tags_prev:
                for t2 in tags:
                    transition[tag2id[t1]][tag2id[t2]] += 1

    _normalize(emission)
    _normalize(transition)

    return emission, transition


if __name__ == '__main__':
    path = '../WSJ-2-12/*/*.POS'
    docs = glob(path)
    parsed = parse(docs)
    emission, transition = counter(parsed)
    sorted_ = sorted(transition[tag2id['START']].items(), key=operator.itemgetter(1))
    print sorted_
