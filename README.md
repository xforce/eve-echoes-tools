<!-- omit in TOC -->

Eve Echoes Tools
=========================
[![GitHub](https://img.shields.io/github/license/xforce/eve-echoes-tools?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Discord](https://img.shields.io/discord/747940644378640425?style=for-the-badge)](https://discord.gg/XZsxXCN)
[![GitHub language count](https://img.shields.io/github/languages/count/xforce/eve-echoes-tools?style=for-the-badge)]()

A collection of tools to convert, extract and modify the game files of Eve Echoes.


This project is only partially maintained and has experienced some breaking changes in the dependencies. You can find
a list of the latest tested and working dependency version at the end of [INSTALLATION.md](INSTALLATION.md). There is
also a `requirements-lock.txt` file that contains the exact package versions.

The Docker installation is currently broken, if you still want to use docker, you have to fix it.

## Docker Installation

The detailed instructions for a manual installations can be found in .

> The docker script is currently broken because the installation requires some hacky solutions. Please install it
> manually

~~Only a few tools in this repo can be installed on your machine, this is generally intended to be run in a clone.~~</br>
The primary use-case of this for now is to dump all the static data of Eve Echoes into JSON files.

The easiest way this can be done is by using the docker image which has all the required tools installed.</br>
Simply run. (With the latest Eve Echoes XAPK in the local directory named `eve.xapk`)

```
docker run -v$(pwd):/data cookiemagic/evee-tools dump_static /data/eve.xapk /data/staticdata
```

> On windows adjust the Mount accordingly.
> You can build the docker image on windows using PowerShell (in the root directory) with
> ```
> docker build -f docker/Dockerfile -t cookiemagic/evee-tools .
> ```
> To run it, you have to put the `$(pwd):/data` part in quotes when using PowerShell:
> ```
> docker run -v "$(pwd):/data" cookiemagic/evee-tools dump_static /data/eve.xapk /data/staticdata
> ``` 


Now you should have all the static data for Eve Echoes ready to use in `staticdata`. Enjoy.

<br />

## [FSD2JSON](fsd2json)

A simple tool to convert the static data fsd files to json, making it easy to inspect and play with.

## [NeoX Tools](https://github.com/xforce/neox-tools)

Simple tools to interact with the engine files of NeoX as used in Eve Echoes and other NetEase games.

## Installing

> This section does probably also no longer work, please refer to the manual installation page.

All you have to do to build it is clone it and run on of the following:

```
cargo install --path {path to tool}
```

> {path to tool} to be replaced by one of the directories in this repo.
