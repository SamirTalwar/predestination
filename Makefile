SHELL := zsh

PYTHON = python3.6

.PHONY: freeze
freeze: env/bin/python
	env/bin/pip freeze > requirements.txt

env/lib/$(PYTHON)/site-packages: env/bin/python requirements.txt
	env/bin/pip install --requirement=requirements.txt

env/bin/python:
	virtualenv --python=$(PYTHON) env
