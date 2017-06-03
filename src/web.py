#!/usr/bin/env python

import os
import os.path
import sys
from collections import namedtuple

import flask
import flask.templating
import flask_socketio

import options
from life import Life

opts = namedtuple('Options', ['input_file'])(os.environ.get('INPUT_FILE'))
app = flask.Flask(__name__, root_path=os.getcwd())
transports = os.environ.get('TRANSPORTS', 'websocket polling').split()
socketio = flask_socketio.SocketIO(app)


@app.route('/')
def index():
    return flask.templating.render_template(
            'index.html', transports=transports)


@socketio.on('start')
def start(data):
    life = Life.from_file(opts.input_file) \
            if opts.input_file \
            else Life.random(int(data['height']), int(data['width']))
    flask_socketio.emit('generation', life.matrix.tolist())


@socketio.on('next')
def next(grid):
    life = Life(grid)
    life.next()
    flask_socketio.emit('generation', life.matrix.tolist())


def main(args):
    global opts
    opts = options.parse(args)
    socketio.run(app)


if __name__ == "__main__":
    main(sys.argv[1:])
