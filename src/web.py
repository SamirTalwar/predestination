#!/usr/bin/env python

from life import Life

import flask
import flask_socketio
import os
import os.path


app = flask.Flask(__name__, root_path=os.getcwd())
socketio = flask_socketio.SocketIO(app)


@app.route('/')
def index():
    return app.send_static_file('index.html')


@socketio.on('start')
def start(data):
    life = Life.random(int(data['height']), int(data['width']))
    while True:
        flask_socketio.emit('generation', life.matrix.tolist())
        life.next()
        socketio.sleep(0.1)


if __name__ == "__main__":
    socketio.run(app)
