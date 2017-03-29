#!/usr/bin/env python

import flask
import os
import os.path


app = flask.Flask(__name__, root_path=os.getcwd())


@app.route('/')
def index():
    return app.send_static_file('index.html')


if __name__ == "__main__":
    app.run()
