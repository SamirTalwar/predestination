class Life:
    def __init__(self, matrix):
        self.matrix = matrix

    def __repr__(self):
        return 'Life({0!r})'.format(self.matrix)

    def __str__(self):
        return str(self.matrix)
