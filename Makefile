SHELL := zsh

PYTHON = python3.6
SITE_PACKAGES = env/lib/$(PYTHON)/site-packages

PORT ?= 8080

.PHONY: web
web:
	env/bin/gunicorn --bind=0.0.0.0:$(PORT) --pythonpath=src,$(SITE_PACKAGES) --worker-class=eventlet --workers=1 web:app

.PHONY: site-packages
site-packages: $(SITE_PACKAGES)

.PHONY: freeze
freeze: env/bin/python
	env/bin/pip freeze > requirements.txt

$(SITE_PACKAGES): env/bin/python requirements.txt
	env/bin/pip install --requirement=requirements.txt

env/bin/python:
	virtualenv --python=$(PYTHON) env
