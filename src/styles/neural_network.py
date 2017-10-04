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
        theta1, theta2 = pickle.load(f)

    def next(grid):
        a0 = matrices.windows(grid).reshape(grid.size, 9)
        a1 = sigmoid(numpy.dot(a0, theta1))
        a2 = sigmoid(numpy.dot(a1, theta2))
        y = numpy.argmax(a2, axis=1)
        return y.reshape(grid.shape).round().astype(int)

    return next


def train():
    iterations = 50000
    width = 10
    height = 10
    size = width * height
    hidden_layer_size = 9
    categories = 2

    numpy.random.seed(0)
    os.makedirs(training_dir, exist_ok=True)

    life = Life.random(width, height).next(reference)
    X = matrices.windows(life.matrix).reshape(size, 9)
    results = life.next(reference).matrix.reshape(size, 1)
    category_indices = numpy.repeat(
            numpy.matrix([numpy.arange(categories)]), size, axis=0)
    y = (category_indices == results).astype(int)

    theta1 = 2 * numpy.random.random((9, hidden_layer_size)) - 1
    theta2 = 2 * numpy.random.random((hidden_layer_size, categories)) - 1

    print('Training...')
    for i in range(1, iterations + 1):
        a0 = X
        a1 = sigmoid(numpy.dot(a0, theta1))
        a2 = sigmoid(numpy.dot(a1, theta2))

        a2_error = y - a2
        if i % 1000 == 0:
            print('Training error after %d iterations: %f'
                  % (i, numpy.mean(numpy.abs(a2_error))))

        a2_delta = numpy.multiply(a2_error, sigmoid_d(a2))
        a1_error = a2_delta.dot(theta2.T)
        a1_delta = numpy.multiply(a1_error, sigmoid_d(a1))

        theta2 += a1.T.dot(a2_delta)
        theta1 += a0.T.dot(a1_delta)

    weights = (theta1, theta2)
    with open(weights_file, 'wb') as f:
        pickle.dump(weights, f)


if __name__ == '__main__':
    train()
else:
    next = construct_next()
