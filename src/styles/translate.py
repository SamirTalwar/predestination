import numpy

directions = \
    set((x, y) for x in range(-1, 2) for y in range(-1, 2)) - set([(0, 0)])


def next(life):
    neighbours = sum(translate(life.matrix, x, y) for (x, y) in directions)
    life.matrix = (
        ((life.matrix == 1) & (neighbours == 2))
        | (neighbours == 3)
    ).astype(int)


def translate(matrix, x, y):
    return numpy.roll(matrix, (y, x), axis=(0, 1))
