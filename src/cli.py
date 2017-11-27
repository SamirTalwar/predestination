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
        self.show_marker = False

    def run(self):
        curses.noecho()
        curses.cbreak()
        curses.start_color()
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_WHITE, curses.COLOR_WHITE)

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
        self.show_marker = False

        self.display()
        self.life = self.life.next(self.style)
        time.sleep(0.1)

        if self.read() == ' ':
            return self.pause

    def pause(self):
        self.stdscr.nodelay(False)

        self.show_marker = True
        self.display()

        ch = self.read()
        if ch == ' ':
            return self.live
        elif ch == 'KEY_UP':
            self.marker = (self.marker[0], self.marker[1] - 1)
        elif ch == 'KEY_DOWN':
            self.marker = (self.marker[0], self.marker[1] + 1)
        elif ch == 'KEY_LEFT':
            self.marker = (self.marker[0] - 1, self.marker[1])
        elif ch == 'KEY_RIGHT':
            self.marker = (self.marker[0] + 1, self.marker[1])

    def read(self):
        try:
            ch = self.stdscr.getkey()
        except curses.error:
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
            self.width = width
            self.height = height
            self.marker = (int(self.width / 2), int(self.height / 2))
            self.life = Life.random(height - 1, width - 1)

    def display(self):
        self.stdscr.clear()
        for i, line in enumerate(self.life.matrix.tolist()):
            self.stdscr.addstr(i, 0, ''.join(CLI.outputs[n] for n in line),
                               curses.color_pair(1))
        if self.show_marker:
            self.stdscr.addstr(self.marker[1], self.marker[0], ' ',
                               curses.color_pair(2))
        self.stdscr.move(self.height - 1, 0)
        self.stdscr.refresh()


class Quit(Exception):
    pass


if __name__ == '__main__':
    main(sys.argv[1:])
