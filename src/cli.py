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
        CLIRunner(self.input_file, stdscr).run()


class CLIRunner:
    def __init__(self, input_file, stdscr):
        self.input_file = input_file
        self.stdscr = stdscr

    def run(self):
        curses.noecho()
        curses.cbreak()

        self.load()

        mode = self.live
        try:
            while True:
                next_mode = mode()
                if next_mode:
                    mode = next_mode
        except (Quit, KeyboardInterrupt):
            pass

    def live(self):
        self.stdscr.nodelay(True)
        self.display()
        self.life.next()
        time.sleep(0.1)

        if self.read() == ' ':
            return self.pause

    def pause(self):
        self.stdscr.nodelay(False)
        self.display()
        if self.read() == ' ':
            return self.live

    def read(self):
        try:
            ch = self.stdscr.getkey()
        except:
            return

        if ch == 'q':
            raise Quit()
        else:
            return ch

    def load(self):
        if self.input_file:
            with open(self.input_file) as f:
                self.life = Life(numpy.matrix([
                    [CLI.inputs[c] for c in line.strip()]
                    for line in f]))
        else:
            height, width = self.stdscr.getmaxyx()
            height -= 1
            width -= 1
            k = height * width
            self.life = Life(
                numpy.matrix(random.choices([0, 1], k=k))
                .reshape((height, width)))

    def display(self):
        self.stdscr.clear()
        for i, line in enumerate(self.life.matrix.tolist()):
            self.stdscr.addstr(i, 0, ''.join(CLI.outputs[n] for n in line))
        self.stdscr.refresh()


class Quit(Exception):
    pass


if __name__ == '__main__':
    curses.wrapper(CLI(*sys.argv[1:]).run)
