SHELL := zsh -e -u -o pipefail

PYTHON = python3.6
SITE_PACKAGES = env/lib/$(PYTHON)/site-packages
TAG = samirtalwar/predestination

.PHONY: site-packages
site-packages: $(SITE_PACKAGES)

$(SITE_PACKAGES): env/bin/python requirements.txt
	env/bin/pip install --requirement=requirements.txt

env/bin/python:
	virtualenv --python=$(PYTHON) env

.PHONY: freeze
freeze: env/bin/python
	env/bin/pip freeze > requirements.txt

.PHONY: upgrade-dependencies
upgrade-dependencies: env/bin/python
	@ packages=($$(env/bin/pip list --outdated --format=json | jq -r '.[] | .name')); \
	if [[ "$${#packages}" -gt 0 ]]; then \
		env/bin/pip install --upgrade $${packages[@]}; \
		env/bin/pip freeze > requirements.txt; \
	fi

.PHONY: docker-build
docker-build:
	docker build --pull --tag=$(TAG) .

.PHONY: docker-push
docker-push: docker-build
	docker push $(TAG)
