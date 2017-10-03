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
        weights = numpy.matrix(pickle.load(f))

    def next(grid):
        return sigmoid(matrices.windows(grid)
                       .reshape(grid.size, 9)
                       .dot(weights)) \
               .reshape(grid.shape) \
               .round()

    return next


next = construct_next()


def train():
    iterations = 10000
    width = 10
    height = 10
    size = width * height
    numpy.random.seed(1)

    life = Life.random(width, height)
    X = matrices.windows(life.matrix).reshape(size, 9)  # size * 9
    y = life.next(reference).matrix.reshape(size, 1)  # size * 1

    theta1 = 2 * numpy.random.random((9, 1)) - 1  # 9 * 1

    for i in range(iterations):
        a1 = X  # size * 9
        z2 = a1.dot(theta1)  # size * 1
        a2 = sigmoid(z2)  # size * 1

        d2 = y - a2  # size * 1
        theta1_grad = numpy.multiply(d2, sigmoid_d(a2)) / size  # size * 1
        theta1 += a1.T.dot(theta1_grad)  # 9 * 1

    return theta1


def dump():
    with open(weights_file, 'rb') as f:
        print(pickle.load(f))


if __name__ == '__main__':
    os.makedirs(training_dir, exist_ok=True)

    print('Training...')
    weights = train()
    with open(weights_file, 'wb') as f:
        pickle.dump(weights, f)
