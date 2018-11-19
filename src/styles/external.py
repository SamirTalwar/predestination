import subprocess

import numpy


class Style:
    NAME = 'external'

    @staticmethod
    def populate_args(parser):
        parser.add_argument('program')

    def __init__(self, args):
        self.program = args.program

    def next(self, grid):
        current_generation = to_string(grid)
        process = subprocess.run(
                self.program, input=current_generation, stdout=subprocess.PIPE,
                encoding='utf-8')
        return from_string(process.stdout)


def to_string(grid):
    return '\n'.join(''.join(str(cell) for cell in row)
                     for row in grid.tolist())


def from_string(string):
    return numpy.matrix([[int(cell) for cell in row]
                         for row in string.split('\n')
                         if row])
