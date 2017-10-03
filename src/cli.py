#!/usr/bin/env python

import curses
import sys
import time

import options
from life import Life


def main(args):
    curses.wrapper(CLI(options.parse(args)).run)


class CLI:
    outputs = {0: '∙', 1: '█'}

    def __init__(self, options):
        self.options = options

    def run(self, stdscr):
        CLIRunner(self.options, stdscr).run()


class CLIRunner:
    def __init__(self, options, stdscr):
        self.input_file = options.input_file
        self.style = options.style
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
        self.life = self.life.next(self.style)
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
            self.life = Life.from_file(self.input_file)
        else:
            height, width = self.stdscr.getmaxyx()
            self.life = Life.random(height - 1, width - 1)

    def display(self):
        self.stdscr.clear()
        for i, line in enumerate(self.life.matrix.tolist()):
            self.stdscr.addstr(i, 0, ''.join(CLI.outputs[n] for n in line))
        self.stdscr.refresh()


class Quit(Exception):
    pass


if __name__ == '__main__':
    main(sys.argv[1:])
