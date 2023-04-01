HOST = yangyangfu

# define image names
IMAGE_NAME = drlax
TAG_DRLAX_CPU = cpu
TAG_DEBUG = debug

# some dockerfile
DOCKERFILE = Dockerfile
DOCKERFILE_DEBUG = Dockerfile.debug

# build image
build_drlax_cpu:
	docker build -f ${DOCKERFILE} --no-cache --rm -t ${HOST}/${IMAGE_NAME}-${TAG_DRLAX_CPU} .

build_debug:
	docker build -f ${DOCKERFILE_DEBUG} --no-cache --rm -t ${HOST}/${IMAGE_NAME}-${TAG_DEBUG} .