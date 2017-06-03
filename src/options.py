import argparse
import sys


def parse(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description='Serve up the Game of Life.')
    parser.add_argument('--file', dest='input_file', default=None)
    return parser.parse_args(args)
