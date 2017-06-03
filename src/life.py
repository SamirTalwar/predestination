import random

import numpy

directions = \
    set((x, y) for x in range(-1, 2) for y in range(-1, 2)) - set([(0, 0)])

inputs = {
    '.': 0,
    'x': 1,
}


class Life:
    def __init__(self, matrix):
        self.matrix = numpy.matrix(matrix)

    @staticmethod
    def from_file(path):
        with open(path) as f:
            return Life([[inputs[c] for c in line.strip()] for line in f])

    @staticmethod
    def random(height, width):
        k = height * width
        return Life(
            numpy.matrix(random.choices([0, 1], k=k))
            .reshape((height, width)))

    def next(self):
        neighbours = sum(translate(self.matrix, x, y) for (x, y) in directions)
        self.matrix = (
            ((self.matrix == 1) & (neighbours == 2))
            | (neighbours == 3)
        ).astype(int)

    def __repr__(self):
        return 'Life({0!r})'.format(self.matrix)

    def __str__(self):
        return str(self.matrix)


def translate(matrix, x, y):
    return numpy.roll(matrix, (y, x), axis=(0, 1))
