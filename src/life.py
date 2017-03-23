import numpy


directions = \
    set((x, y) for x in range(-1, 2) for y in range(-1, 2)) - set([(0, 0)])


class Life:
    def __init__(self, matrix):
        self.matrix = matrix

    def next(self):
        neighbours = sum(translate(self.matrix, x, y) for (x, y) in directions)
        self.matrix = numpy.logical_or(
            numpy.logical_and(self.matrix == 1, neighbours == 2),
            neighbours == 3)

    def __repr__(self):
        return 'Life({0!r})'.format(self.matrix)

    def __str__(self):
        return str(self.matrix)


def translate(matrix, x, y):
    return numpy.roll(numpy.roll(matrix, x, axis=1), y, axis=0)
