#!/bin/bash

if docker buildx &> /dev/null; then
    ( cd .. && docker buildx build --load --platform=amd64,arm64 -f docker/Dockerfile -t cookiemagic/evee-tools . )
else
    ( cd .. && docker build -f docker/Dockerfile -t cookiemagic/evee-tools . )
fi
