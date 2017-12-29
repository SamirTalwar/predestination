SHELL := zsh -e -u -o pipefail

CONDA_ENVIRONMENT = predestination
CONDA_FILE = conda.txt
TAG = samirtalwar/predestination

.PHONY: conda-save
conda-save:
	conda list --name=$(CONDA_ENVIRONMENT) --explicit > $(CONDA_FILE)

.PHONY: conda-load
conda-load:
	conda create --name=$(CONDA_ENVIRONMENT) --file=$(CONDA_FILE)

.PHONY: conda-update
conda-update:
	conda update --name=$(CONDA_ENVIRONMENT) --all
	conda list --name=$(CONDA_ENVIRONMENT) --explicit > $(CONDA_FILE)

.PHONY: docker-build
docker-build:
	docker build --pull --tag=$(TAG) .

.PHONY: docker-push
docker-push: docker-build
	docker push $(TAG)
