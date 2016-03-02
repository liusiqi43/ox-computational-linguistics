import random
import numpy as np
import matplotlib.pyplot as plt

from tsne import bh_sne

def tsne(embedding, word_2_id, sample_size = 1000):
    embedding_2d = bh_sne(embedding.astype(np.float64))
    keys = random.sample(word_2_id.keys(), sample_size)

    fig, ax = plt.subplots()
    for k in keys:
        id = word_2_id[k]
        ax.annotate(k, (embedding_2d[id, 0], embedding_2d[id, 1]))
    plt.show()
