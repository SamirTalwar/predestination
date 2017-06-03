#!/usr/bin/env python

import os
import os.path

import flask
import flask.templating
import flask_socketio

from life import Life

app = flask.Flask(__name__, root_path=os.getcwd())
transports = os.environ.get('TRANSPORTS', 'websocket polling').split()
socketio = flask_socketio.SocketIO(app)


@app.route('/')
def index():
    return flask.templating.render_template(
            'index.html', transports=transports)


@socketio.on('start')
def start(data):
    life = Life.random(int(data['height']), int(data['width']))
    flask_socketio.emit('generation', life.matrix.tolist())


@socketio.on('next')
def next(grid):
    life = Life(grid)
    life.next()
    flask_socketio.emit('generation', life.matrix.tolist())


if __name__ == "__main__":
    socketio.run(app)
