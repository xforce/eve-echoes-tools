#!/bin/bash

docker buildx &> /dev/null
status=$?

if [ "$status" -eq 0 ] &> /dev/null && [ "$1" == "push" ]; then
    ( cd .. && docker buildx build --push --platform=amd64,arm64 -f docker/Dockerfile -t cookiemagic/evee-tools . )
else
    ( cd .. && docker build -f docker/Dockerfile -t cookiemagic/evee-tools . )
fi
