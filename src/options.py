import argparse
import importlib
import os
import os.path
import sys
from collections import namedtuple

STYLES_DIR = os.path.join(os.path.dirname(__file__), 'styles')
STYLE_NAMES = [os.path.splitext(f)[0]
               for f in os.listdir(STYLES_DIR)
               if os.path.isfile(os.path.join(STYLES_DIR, f))
               and os.path.splitext(f)[1] == '.py']
STYLES = {name.replace('_', '-'):
          importlib.import_module('styles.' + name).Style
          for name in STYLE_NAMES}
DEFAULT_STYLE_NAME = 'translate'

Options = namedtuple('Options', ['style', 'input_file'])


def parse(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description='Serve up the Game of Life.')
    parser.add_argument('--file', dest='input_file')
    subparsers = parser.add_subparsers(dest='style')
    for (name, style) in STYLES.items():
        subparser = subparsers.add_parser(name)
        if hasattr(style, 'populate_args'):
            style.populate_args(subparser)
    result = parser.parse_args(args)
    style = STYLES[result.style]
    return Options(style(result), result.input_file)
