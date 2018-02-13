SHELL := zsh -e -u -o pipefail

TAG = samirtalwar/predestination

.PHONY: conda-env-create
conda-env-create:
	conda env create

.PHONY: conda-env-update
conda-env-update:
	conda env update

.PHONY: docker-build
docker-build:
	docker build --pull --tag=$(TAG) .

.PHONY: docker-push
docker-push: docker-build
	docker push $(TAG)
