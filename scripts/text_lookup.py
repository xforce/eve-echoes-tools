#!/usr/bin/env python
import mmh3
import argparse
import json

parser = argparse.ArgumentParser(
    description='Lookup a translated from the chinese source text')
parser.add_argument('source', action='store',
                    help="The source chinese text you want to get translated into a different language")
parser.add_argument('lang', action='store',
                    help="The target language you want the Chinese text to get translated to")

args = parser.parse_args()

with open("staticdata/gettext/msg_index/index.json") as msg_index:
    msg_index = json.load(msg_index)
    key = mmh3.hash(args.source, signed=False, seed=2538058380)
    msg_id = msg_index[str(key)]
    category_id = msg_id / 1000
    data_path = 'staticdata/gettext/%s/%d.json' % (args.lang, category_id)
    with open(data_path) as localization:
        localization = json.load(localization)
        text = localization[str(msg_id)]
        print(text)
