import numpy


class Style:
    NAME = 'translate'

    def __init__(self, args):
        pass

    def next(self, grid):
        neighbours = sum(translate(grid, x, y) for (x, y) in directions)
        return (
            ((grid == 1) & (neighbours == 2))
            | (neighbours == 3)
        ).astype(int)


def translate(matrix, x, y):
    return numpy.roll(matrix, (y, x), axis=(0, 1))


directions = \
    set((x, y) for x in range(-1, 2) for y in range(-1, 2)) - set([(0, 0)])
