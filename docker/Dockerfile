FROM alpine:edge as cargo-build

RUN apk add -X http://dl-cdn.alpinelinux.org/alpine/edge/testing rust cargo

WORKDIR /app
ADD Cargo.toml Cargo.toml
# ADD Cargo.lock Cargo.lock
ADD fsd2json/src /app/fsd2json/src
ADD fsd2json/Cargo.toml /app/fsd2json/Cargo.toml

ADD neox-tools/src /app/neox-tools/src
ADD neox-tools/Cargo.toml /app/neox-tools/Cargo.toml

RUN cargo install --path fsd2json --root /usr/local/
RUN cargo install --path neox-tools --root /usr/local/

FROM alpine:edge

COPY --from=cargo-build /usr/local/bin/fsd2json /usr/local/bin/fsd2json
COPY --from=cargo-build /usr/local/bin/npktool /usr/local/bin/npktool

RUN apk add --no-cache -X http://dl-cdn.alpinelinux.org/alpine/edge/testing git python3 py3-pip py3-mmh3 bash
RUN pip install --no-cache-dir mmh3 xdis spark-parser parso

RUN apk add --no-cache -X http://dl-cdn.alpinelinux.org/alpine/edge/testing libgcc
ADD https://api.github.com/repos/xforce/eve-echoes-tools/compare/main...HEAD /dev/null
WORKDIR /opt/eve-echoes-tools
COPY . /opt/eve-echoes-tools

COPY docker/cmd.sh /opt/
ENTRYPOINT ["/opt/cmd.sh"]
