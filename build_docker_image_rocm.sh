#!/bin/bash

export TAG=1
export BASE_IMAGE=rocm/dev-ubuntu-20.04:5.2.3
export BUILD_DOCKER_FILE=ci/docker/build.Dockerfile
export COMPILER=clang@10.0.0
export CXXSTD=17
export DEPLOY_DOCKER_FILE=ci/docker/deploy-amdgpu.Dockerfile
export DEPLOY_BASE_IMAGE=rocm/dev-ubuntu-20.04:5.2.3
export BUILD_IMAGE=release-amdgpu-clang10/build
export DEPLOY_IMAGE=release-amdgpu-clang10-test
export SLURM_CONSTRAINT=mc
export SPACK_SHA=08efe13efb627793fbe048542890ee85668c8100
export SPACK_DLAF_REPO=./spack
export SPACK_ENVIRONMENT=ci/docker/amdgpu-release-rocm523.yaml
export EXTRA_APTGET_DEPLOY=""
export USE_MKL="OFF"
export USE_ROCBLAS="ON"

#docker build -t $DEPLOY_IMAGE --build-arg BUILD_IMAGE=$BUILD_IMAGE:$TAG --build-arg DEPLOY_BASE_IMAGE --build-arg EXTRA_APTGET_DEPLOY --build-arg USE_MKL --build-arg USE_ROCBLAS -f $DEPLOY_DOCKER_FILE --network=host .
docker build -t $BUILD_IMAGE:$TAG -t $BUILD_IMAGE:latest --cache-from $BUILD_IMAGE:latest --build-arg BASE_IMAGE --build-arg BUILDKIT_INLINE_CACHE=1 --build-arg SPACK_SHA --build-arg EXTRA_APTGET --build-arg COMPILER --build-arg CXXSTD --build-arg SPACK_ENVIRONMENT --build-arg SPACK_DLAF_REPO --build-arg USE_MKL -f $BUILD_DOCKER_FILE --network=host .
