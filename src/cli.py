#!/usr/bin/env python

from life import Life

import curses
import numpy
import sys
import time


class CLI:
    inputs = {'.': 0, 'x': 1}
    outputs = {0: '.', 1: 'x'}

    def __init__(self, input_file):
        with open(input_file) as f:
            self.life = Life(numpy.matrix([
                [CLI.inputs[c] for c in line.strip()]
                for line in f]))

    def run(self, stdscr):
        try:
            while True:
                self.display(stdscr)
                time.sleep(0.5)
                self.life.next()
        except KeyboardInterrupt:
            pass

    def display(self, stdscr):
        stdscr.clear()
        for i, line in enumerate(self.life.matrix.tolist()):
            stdscr.addstr(i, 0, ''.join(CLI.outputs[n] for n in line))
        stdscr.refresh()


if __name__ == '__main__':
    curses.wrapper(CLI(sys.argv[1]).run)
