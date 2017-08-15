SHELL := zsh

PYTHON = python3.6
SITE_PACKAGES = env/lib/$(PYTHON)/site-packages

.PHONY: site-packages
site-packages: $(SITE_PACKAGES)

$(SITE_PACKAGES): env/bin/python requirements.txt
	env/bin/pip install --requirement=requirements.txt

env/bin/python:
	virtualenv --python=$(PYTHON) env

.PHONY: freeze
freeze: env/bin/python
	env/bin/pip freeze > requirements.txt
