SHELL := zsh -e -u -o pipefail

TAG = samirtalwar/predestination

.PHONY: docker-build
docker-build:
	docker build --pull --tag=$(TAG) --file=build.Dockerfile .

.PHONY: docker-push
docker-push: docker-build
	docker push $(TAG)

.PHONY: docker-run
docker-run: docker-build
	docker run --rm --interactive --tty --publish=8080:8080 $(TAG)
