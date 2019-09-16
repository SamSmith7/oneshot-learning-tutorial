import random
import numpy as np


def get_batch(batch_size, X, y, c):

    n_classes, n_examples, w, h = X.shape

    # needs to pick some random classes 
    categories = random.choice(n_classes, size = (batch_size,), replace = False)
    pairs = [np.zeros((batch_size, h, w, 1)) for i in range(2)]
    targets = np.zeros((batch_size,))

    targets[batch_size//2:] = 1

    for i in range(batch_size):
        category = categories[i]
        idx_1 = random.randint(0, n_examples)
        pairs[0][i,:,:,:] = X[category, idx_1].reshape(w, h, 1)

        idx_2 = random.randint(0, n_examples)

        if i >= batch_size // 2:
            category_2 = category
        else:
            category_2 = (category + random.randint(1, n_classes)) % n_classes

        pairs[1][i,:,:,:] = X[category_2, idx_2].reshape(w, h, 1)

    return pairs, targets

def generate(batch_size, X, y, c):

    while True:
        pairs, targets = get_batch(batch_size, s)
        yield (pairs, targets)
