import numpy


class Life:
    def __init__(self, matrix):
        self.matrix = matrix

    def next(self):
        self.matrix = numpy.roll(numpy.roll(self.matrix, 1, axis=0), 1, axis=1)

    def __repr__(self):
        return 'Life({0!r})'.format(self.matrix)

    def __str__(self):
        return str(self.matrix)
