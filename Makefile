.PHONY: build
build:
	script/build

.PHONY: push
push: build
	docker push r8.im/username/my-model

.PHONY: predict
predict:
	replicate run username/my-model prompt=horse
