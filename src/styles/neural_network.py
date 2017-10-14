import itertools
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
        weights = pickle.load(f)

    def next(grid):
        neurons = [matrices.windows(grid).reshape(grid.size, 9)]
        for layer_weights in weights:
            neurons.append(sigmoid(neurons[-1] * layer_weights))
        y = numpy.argmax(neurons[-1], axis=1)
        return y.reshape(grid.shape).round().astype(int)

    return next


def sliding_window(values, window_size):
    lists = (itertools.islice(values, i, None) for i in range(window_size))
    return zip(*lists)


def random_weights(rows, columns):
    return numpy.matrix(numpy.random.random((rows, columns)) * 2 - 1)


def training_data(width, height):
    categories = numpy.array([0, 1])
    size = width * height
    category_indices = numpy.repeat(numpy.matrix([categories]), size, axis=0)

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
        life = Life.random(width, height)
        life, X_and_y_1 = iterate(life)
        life, X_and_y_2 = iterate(life)
        X_and_y = numpy.concatenate((X_and_y, X_and_y_1, X_and_y_2), axis=0)

    X = X_and_y[:, :9]
    y = X_and_y[:, 9:]
    return (X, y)


def train():
    iterations = 100000
    width = 10
    height = 10
    hidden_layers = [25]
    learning_rate = 0.01

    numpy.random.seed(1)
    X, y = training_data(width, height)

    columns = [X.shape[1]] + hidden_layers + [y.shape[1]]
    weights = [random_weights(a, b) for (a, b) in sliding_window(columns, 2)]

    print('Training...')
    for i in range(1, iterations + 1):
        neurons = [X]
        for layer_weights in weights:
            neurons.append(sigmoid(neurons[-1] * layer_weights))

        error = y - neurons[-1]
        next_layer_error = error
        weight_deltas = []
        for layer_weights, (next_layer, previous_layer) \
                in zip(reversed(weights),
                       sliding_window(list(reversed(neurons)), 2)):
            d = numpy.multiply(next_layer_error, sigmoid_d(next_layer))
            weight_delta = previous_layer.T * d * learning_rate
            weight_deltas.append(weight_delta)
            next_layer_error = d * layer_weights.T
        weight_deltas.reverse()

        weights = [layer_weights + delta
                   for (layer_weights, delta)
                   in zip(weights, weight_deltas)]

        if i % 1000 == 0:
            print('Training error after %d iterations: %f'
                  % (i, numpy.mean(numpy.abs(error))))

    with open(weights_file, 'wb') as f:
        pickle.dump(weights, f)
    print('Saved.')


if __name__ == '__main__':
    train()
else:
    next = construct_next()
