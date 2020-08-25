FROM debian:bullseye

RUN apt update; apt install -y curl
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

RUN apt update; apt install -y git g++ python2 python2-dev
RUN curl https://bootstrap.pypa.io/get-pip.py --output get-pip.py
RUN python2 get-pip.py
RUN apt clean \
    && rm -rf /var/lib/apt/lists/*

COPY cmd.sh /opt/

RUN pip2 install mmh3 rotor xdis spark-parser

RUN git clone --recursive https://github.com/xforce/eve-echoes-tools.git /opt/eve-echoes-tools

ENTRYPOINT ["/opt/cmd.sh"]

