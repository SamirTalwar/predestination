#!/usr/bin/env python

from life import Life

import curses
import numpy
import random
import sys
import time


class CLI:
    inputs = {'.': 0, 'x': 1}
    outputs = {0: '∙', 1: '█'}

    def __init__(self, input_file=None):
        self.input_file = input_file

    def run(self, stdscr):
        self.load(stdscr)
        try:
            while True:
                self.display(stdscr)
                time.sleep(0.1)
                self.life.next()
        except KeyboardInterrupt:
            pass

    def load(self, stdscr):
        if self.input_file:
            with open(self.input_file) as f:
                self.life = Life(numpy.matrix([
                    [CLI.inputs[c] for c in line.strip()]
                    for line in f]))
        else:
            height, width = stdscr.getmaxyx()
            height -= 1
            width -= 1
            k = height * width
            self.life = Life(
                numpy.matrix(random.choices([0, 1], k=k))
                .reshape((height, width)))

    def display(self, stdscr):
        stdscr.clear()
        for i, line in enumerate(self.life.matrix.tolist()):
            stdscr.addstr(i, 0, ''.join(CLI.outputs[n] for n in line))
        stdscr.refresh()


if __name__ == '__main__':
    curses.wrapper(CLI(*sys.argv[1:]).run)
