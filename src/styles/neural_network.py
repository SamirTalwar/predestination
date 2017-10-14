import os
import os.path
import pickle

import numpy

import matrices
import styles.translate as reference
from life import Life

root = os.path.realpath(os.path.join(
    os.path.dirname(__file__), os.path.pardir, os.path.pardir))
training_dir = os.path.join(root, 'test', 'training')
weights_file = os.path.join(training_dir, 'weights.pickle')


def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))


def sigmoid_d(x):
    return numpy.multiply(x, 1 - x)


def construct_next():
    with open(weights_file, 'rb') as f:
        w1, w2 = pickle.load(f)

    def next(grid):
        a0 = matrices.windows(grid).reshape(grid.size, 9)
        a1 = sigmoid(a0 * w1)
        a2 = sigmoid(a1 * w2)
        y = numpy.argmax(a2, axis=1)
        return y.reshape(grid.shape).round().astype(int)

    return next


def random_weights(rows, columns):
    return numpy.matrix(numpy.random.random((rows, columns)) * 2 - 1)


def training_data(width, height, categories):
    size = width * height
    category_indices = numpy.repeat(
            numpy.matrix([numpy.arange(categories)]), size, axis=0)

    os.makedirs(training_dir, exist_ok=True)

    def iterate(life):
        X = matrices.windows(life.matrix).reshape(size, 9)
        next_life = life.next(reference)
        results = next_life.matrix.reshape(size, 1)
        y = (category_indices == results).astype(int)
        return (next_life, numpy.concatenate((X, y), axis=1))

    life = Life.random(width, height)
    life, X_and_y = iterate(life)
    for i in range(9):
        life, new_X_and_y = iterate(life)
        X_and_y = numpy.concatenate((X_and_y, new_X_and_y), axis=0)

    X_and_y = numpy.unique(X_and_y, axis=0)
    X = X_and_y[:, :9]
    y = X_and_y[:, 9:]
    return (X, y)


def train():
    iterations = 100000
    width = 10
    height = 10
    categories = 2
    hidden_layer_size = 25
    learning_rate = 0.1

    numpy.random.seed(1)
    X, y = training_data(width, height, categories)

    w1 = random_weights(X.shape[1], hidden_layer_size)
    w2 = random_weights(hidden_layer_size, y.shape[1])

    print('Training...')
    for i in range(1, iterations + 1):
        a0 = X
        a1 = sigmoid(a0 * w1)
        a2 = sigmoid(a1 * w2)

        a2_error = y - a2
        d2 = numpy.multiply(a2_error, sigmoid_d(a2))
        w2_delta = a1.T * d2 * learning_rate
        d1 = numpy.multiply(d2 * w2.T, sigmoid_d(a1))
        w1_delta = a0.T * d1 * learning_rate

        w1 += w1_delta
        w2 += w2_delta

        if i % 1000 == 0:
            print('Training error after %d iterations: %f'
                  % (i, numpy.mean(numpy.abs(a2_error))))

    weights = (w1, w2)
    with open(weights_file, 'wb') as f:
        pickle.dump(weights, f)
    print('Saved.')


if __name__ == '__main__':
    train()
else:
    next = construct_next()
