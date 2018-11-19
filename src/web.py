#!/usr/bin/env python

import os
import os.path
import sys

import flask
import flask.templating
import flask_socketio

import options
from life import Life

opts = None
app = flask.Flask(__name__, root_path=os.getcwd())
transports = os.environ.get("TRANSPORTS", "websocket polling").split()
socketio = flask_socketio.SocketIO(app)


@app.route("/")
def index():
    return flask.templating.render_template("index.html", transports=transports)


@socketio.on("start")
def start(data):
    life = (
        Life.from_file(opts.input_file)
        if opts.input_file
        else Life.random(int(data["height"]), int(data["width"]))
    )
    flask_socketio.emit("generation", life.matrix.tolist())


@socketio.on("next")
def next(grid):
    life = Life(grid)
    life = life.step(opts.style)
    flask_socketio.emit("generation", life.matrix.tolist())


if __name__ == "__main__":
    socketio.run(app)
else:
    args_offset = sys.argv.index("--") + 1
    args = sys.argv[args_offset:] if "--" in sys.argv else sys.argv[1:]
    opts = options.parse(args)
