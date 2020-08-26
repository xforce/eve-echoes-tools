<!-- omit in TOC -->

Eve Echoes Tools
=========================
[![GitHub](https://img.shields.io/github/license/xforce/eve-echoes-tools?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Discord](https://img.shields.io/discord/747940644378640425?style=for-the-badge)](https://discord.gg/XZsxXCN)
[![GitHub language count](https://img.shields.io/github/languages/count/xforce/eve-echoes-tools?style=for-the-badge)]()
[![Data Download](https://img.shields.io/badge/Data-v1.5.5-yellow?style=for-the-badge)](http://eve-echoes-data.s3-website.eu-central-1.amazonaws.com/v1.5.5/)

A collection of tools to convert, extract and modify the game files of Eve Echoes.

<br>

## Installation

Only a few tools in this repo can be installed on your machine, this is generally intended to be run in a clone.</br>
The primary use-case of this for now is to dump all the static data of Eve Echoes into JSON files.

The easiest way this can be done is by using the docker image which has all the required tools installed.</br>
Simply run. (With the latest Eve Echoes XAPK in the local directory named `eve.xapk`)

```
docker run -v$(pwd):/data cookiemagic/evee-tools dump_static /data/eve.xapk /data/staticdata
```

> On windows adjust the Mount accordingly.

Now you should have all the static data for Eve Echoes ready to use in `staticdata`. Enjoy.

<br />

Alternatively all the static data is provided for your convience in the following S3 Bucket.

[Browsable version from the web](http://eve-echoes-data.s3-website.eu-central-1.amazonaws.com/)

[Direct S3 Link](https://s3-eu-central-1.amazonaws.com/eve-echoes-data)
> The S3 Bucket was last updated on 08/25/20 with v1.5.5 data
<br />
<br />

## [FSD2JSON](fsd2json)

A simple tool to convert the static data fsd files to json, making it easy to inspect and play with.

## [NeoX Tools](https://github.com/xforce/neox-tools)

Simple tools to interact with the engine files of NeoX as used in Eve Echoes and other NetEase games.

## Installing

All you have to do to build it is clone it an run on of the following:

```
cargo install --path <path to tool>
```

> <Path to tool> to be replaced by one of the directories in this repo.
