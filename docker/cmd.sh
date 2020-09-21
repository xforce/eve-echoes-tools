#!/bin/ash

export PATH=$PATH:/opt/eve-echoes-tools/scripts
export PATH="$HOME/.cargo/bin:$PATH"

cd /opt/eve-echoes-tools

if [ $1 == "dump_static" ]; then
python3 scripts/dump_static_data.py $2 $3
fi

