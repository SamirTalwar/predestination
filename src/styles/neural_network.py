import os
import os.path
import pickle

import numpy

import matrices
import styles.translate as reference_style
from life import Life

root = os.path.realpath(os.path.join(
    os.path.dirname(__file__), os.path.pardir, os.path.pardir))
training_dir = os.path.join(root, 'test', 'training')
weights_file = os.path.join(training_dir, 'weights.pickle')

parameters = {
    'iterations': 100000,
    'width': 10,
    'height': 10,
    'hidden_layers': [25],
    'learning_rate': 0.01,
    'random_seed': 1,
}


class Style:
    @staticmethod
    def populate_args(parser):
        parser.add_argument('--weight-file')

    def __init__(self, args):
        with open(args.weight_file or weights_file, 'rb') as f:
            self.weights = pickle.load(f)

    def next(self, grid):
        current_layer = [matrices.windows(grid).reshape(grid.size, 9)]
        for layer_weights in self.weights:
            current_layer = sigmoid(current_layer * layer_weights)
        y = numpy.argmax(current_layer, axis=1)
        return y.reshape(grid.shape).round().astype(int)


def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))


def sigmoid_d(x):
    return numpy.multiply(x, 1 - x)


def pairwise(values):
    return zip(values, values[1:])


def random_weights(rows, columns):
    return numpy.matrix(numpy.random.random((rows, columns)) * 2 - 1)


def training_data(width, height):
    categories = numpy.array([0, 1])
    input_columns = 9
    size = width * height
    category_indices = numpy.repeat(numpy.matrix([categories]), size, axis=0)

    os.makedirs(training_dir, exist_ok=True)

    reference = reference_style.Style(None)

    def iterate(life):
        X = matrices.windows(life.matrix).reshape(size, input_columns)
        next_life = life.next(reference)
        results = next_life.matrix.reshape(size, 1)
        y = (category_indices == results).astype(int)
        return (next_life, numpy.concatenate((X, y), axis=1))

    X_and_y = numpy.matrix(numpy.zeros((0, input_columns + categories.size)))
    for i in range(10):
        life = Life.random(width, height)
        life, X_and_y_1 = iterate(life)
        life, X_and_y_2 = iterate(life)
        X_and_y = numpy.concatenate((X_and_y, X_and_y_1, X_and_y_2), axis=0)

    X = X_and_y[:, :input_columns]
    y = X_and_y[:, input_columns:]
    return (X, y)


def train(
        iterations, width, height, hidden_layers, learning_rate, random_seed):
    numpy.random.seed(random_seed)
    X, y = training_data(width, height)

    columns = [X.shape[1]] + hidden_layers + [y.shape[1]]
    weights = [random_weights(a, b) for (a, b) in pairwise(columns)]

    print('Training...')
    for i in range(1, iterations + 1):
        neurons = [X]
        for layer_weights in weights:
            neurons.append(sigmoid(neurons[-1] * layer_weights))

        backpropagation = reversed(list(zip(weights, pairwise(neurons))))
        error = y - neurons[-1]
        next_layer_error = error
        weight_deltas = []
        for layer_weights, (previous_layer, next_layer) in backpropagation:
            d = numpy.multiply(next_layer_error, sigmoid_d(next_layer))
            weight_delta = previous_layer.T * d * learning_rate
            weight_deltas.insert(0, weight_delta)
            next_layer_error = d * layer_weights.T

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
    train(**parameters)
