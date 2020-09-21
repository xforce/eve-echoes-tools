FROM alpine:edge

RUN apk add --no-cache -X http://dl-cdn.alpinelinux.org/alpine/edge/testing git python3 rustup rust cargo py3-pip py3-mmh3
RUN pip install --no-cache-dir mmh3 xdis spark-parser parso

ADD https://api.github.com/repos/xforce/eve-echoes-tools/compare/main...HEAD /dev/null
RUN git clone --recursive https://github.com/xforce/eve-echoes-tools.git /opt/eve-echoes-tools

WORKDIR /opt/eve-echoes-tools/
RUN cargo install --path fsd2json; cargo clean
RUN cargo install --path neox-tools; cargo clean
WORKDIR /

COPY cmd.sh /opt/
ENTRYPOINT ["/opt/cmd.sh"]
