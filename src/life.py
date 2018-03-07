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
            numpy.matrix(numpy.random.random_integers(0, 1, k))
            .reshape((height, width)))

    def next(self, style):
        return Life(style.next(self.matrix))

    @property
    def height(self):
        return self.matrix.shape[0]

    @property
    def width(self):
        return self.matrix.shape[1]

    def __repr__(self):
        return 'Life({0!r})'.format(self.matrix)

    def __str__(self):
        return str(self.matrix)
