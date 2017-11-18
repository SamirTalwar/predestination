import argparse
import importlib
import sys
from collections import namedtuple

DEFAULT_STYLE_NAME = 'translate'

Options = namedtuple('Options', ['style', 'input_file'])


def parse(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description='Serve up the Game of Life.')
    parser.add_argument('--style')
    parser.add_argument('--file', dest='input_file')
    result = parser.parse_args(args)
    return process(result.style, result.input_file)


def process(style_name, input_file):
    style_module = 'styles.' + (style_name or DEFAULT_STYLE_NAME)
    style = importlib.import_module(style_module)
    return Options(style.Style(), input_file)
