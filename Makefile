SHELL := zsh -e -u -o pipefail

TAG = samirtalwar/predestination

.PHONY: docker-build
docker-build:
	docker build --pull --tag=$(TAG) .

.PHONY: docker-push
docker-push: docker-build
	docker push $(TAG)
