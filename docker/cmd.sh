#!/bin/bash

export PATH=$PATH:/opt/eve-echoes-tools/scripts
export PATH="$HOME/.cargo/bin:$PATH"

OPTIND=1
while getopts ":c:" opt; do
  case ${opt} in
    c )
        # crypt_plugin="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
        crypt_plugin="${OPTARG}"
        ;;
   \? )
     break
     ;;
  esac
done
shift $((OPTIND -1))

if [ ! -z "$crypt_plugin" ]
then
cp $crypt_plugin /opt/eve-echoes-tools/neox-tools/scripts/script_redirect_plug.py
fi

subcommand=$1; shift

case "$subcommand" in
  dump_static)
    OPTIND=1
    while getopts ":p-:" opt; do
      case ${opt} in
        - )
         case "${OPTARG}" in
            patch )
                patch="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
         esac;;
        p )
          patch="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
          ;;
        \? )
          echo "Invalid Option: -$OPTARG" 1>&2
          exit 1
          ;;
        : )
          echo "Invalid Option: -$OPTARG requires an argument" 1>&2
          exit 1
          ;;
      esac
    done
    patch_path=$patch
    xapk_path="${!OPTIND}"
    OPTIND=$(( $OPTIND + 1 ))
    out_dir="${!OPTIND}"
    OPTIND=$(( $OPTIND + 1 ))

    echo $patch_path
    echo $xapk_path
    echo $out_dir

    pushd /opt/eve-echoes-tools
    if [ -z "$patch_path" ]
    then 
        python3 scripts/dump_static_data.py --xapk $xapk_path $out_dir  --auto
    else
        python3 scripts/dump_static_data.py -p $patch_path --xapk $xapk_path $out_dir --auto
    fi
    popd

    ;;
esac

