# Migh want to replace this also with python:3.10-alpine, but we don't really need python here
# and for the moment two different images don't seem to cause any problems.
FROM alpine:edge as cargo-build

# Dependencies for neox-tools/fsd2json
RUN apk add -X http://dl-cdn.alpinelinux.org/alpine/edge/testing rust cargo
# Dependencies for unnpk
RUN apk add --no-cache libmagic file file-dev zlib-dev make

WORKDIR /app
ADD Cargo.toml Cargo.toml
# ADD Cargo.lock Cargo.lock
ADD fsd2json/src /app/fsd2json/src
ADD fsd2json/Cargo.toml /app/fsd2json/Cargo.toml

ADD neox-tools/src /app/neox-tools/src
ADD neox-tools/Cargo.toml /app/neox-tools/Cargo.toml

ADD neox-tools/scripts/unnpk /app/unnpk/src

# Building unnpk
WORKDIR /app/unnpk/src/
RUN make

# Build neox-tools/fsd2json
WORKDIR /app
RUN cargo install --path fsd2json --root /usr/local/
RUN cargo install --path neox-tools --root /usr/local/



FROM python:3.10-alpine
# We need python3.10, xdis does not support python 3.11 at the moment

COPY --from=cargo-build /usr/local/bin/fsd2json /usr/local/bin/fsd2json
COPY --from=cargo-build /usr/local/bin/npktool /usr/local/bin/npktool
COPY --from=cargo-build /app/unnpk/src /opt/eve-echoes-tools/neox-tools/scripts/unnpk

RUN apk add --no-cache -X http://dl-cdn.alpinelinux.org/alpine/edge/testing git python3 py3-pip py3-mmh3 bash

RUN apk add --no-cache -X http://dl-cdn.alpinelinux.org/alpine/edge/testing libgcc

ADD https://api.github.com/repos/xforce/eve-echoes-tools/compare/main...HEAD /dev/null
WORKDIR /opt/eve-echoes-tools
COPY . /opt/eve-echoes-tools

RUN pip install --break-system-packages --no-cache-dir -r requirements.txt

# Apply xdis patch
# This is a very hacky solution, it will edit the library file of xdis
RUN python scripts/docker_fix_xdis.py

COPY docker/cmd.sh /opt/
ENTRYPOINT ["/opt/cmd.sh"]
