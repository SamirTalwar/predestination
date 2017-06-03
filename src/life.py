import random

import numpy

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

    def __repr__(self):
        return 'Life({0!r})'.format(self.matrix)

    def __str__(self):
        return str(self.matrix)
