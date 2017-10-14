# Predestination

Several implementations of [Conway's Game of Life][].

[Conway's Game of Life]: https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life

## Building

It's Python, so all you need is Python 3.6 and [virtualenv][] (which you can install with `pip install virtualenv` or similar).

Then just run `make` to create an environment and all the application dependencies.

[virtualenv]: https://virtualenv.pypa.io/

## Running

It's probably easiest to start with the CLI.

### Running as a command-line application

Just run `./cli`. Run `./cli --help` for options.

### Running as a web application

Just run `./web`. It accepts the same options as `./cli`, and you can set the `PORT` environment variable to change the port.

### Styles

You can configure the application to use a different style of iteration.

There are currently three styles:

  * `translate` (the default) uses matrix translation to calculate the next generation.
  * `mapping` uses a huge lookup table.
  * `neural_network` uses a neural network, which requires training first.
    To train it, run `PYTHONPATH=src python ./src/styles/neural_network.py`.

### Input

By default, Predestination generates a random, uniformly-distributed game the same size as your window, and then iterates from there. To use a file as the starting game instead, just pass it using the `--file` option. You can see an example in *test/fixtures/glider.life*.
