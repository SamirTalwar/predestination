# Predestination

Several implementations of [Conway's Game of Life][].

[![Recording](https://asciinema.org/a/142461.png)](https://asciinema.org/a/142461)

[Conway's Game of Life]: https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life

## Building

This project uses Python 3.6 and [`pipenv`][pipenv].

First, install the dependencies with `pipenv install`.

Then just run `make` to create an environment and all the application dependencies.

[pipenv]: https://docs.pipenv.org/

## Running

It's probably easiest to start with the CLI.

### Running as a command-line application

Just run `./cli`. Run `./cli --help` for options.

### Running as a web application

Just run `./web`. It accepts the same options as `./cli`, and you can set the `PORT` environment variable to change the port.

### Styles

You can configure the application to use a different style of iteration.

There are currently three styles, which can be specified on the command line (e.g. `./cli neural-network`).

  * `translate` (the default) uses matrix translation to calculate the next generation.
  * `mapping` uses a huge lookup table.
  * `neural-network` uses a basic 3-layer neural network, which requires training first.
    To train it, run `PYTHONPATH=src python ./src/styles/neural_network.py`.

### Input

By default, Predestination generates a random, uniformly-distributed game the same size as your window, and then iterates from there. To use a file as the starting game instead, just pass it using the `--file` option. You can see an example in *test/fixtures/glider.life*.
