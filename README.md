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

The detailed instructions for a manual installations can be found in [INSTALLATION.md](INSTALLATION.md).

Only a few tools in this repo can be installed on your machine. ~~this is generally intended to be run in a clone.~~</br>
The primary use-case of this for now is to dump all the static data of Eve Echoes into JSON files.

The easiest way this can be done is by using the docker image which has all the required tools installed.</br>
Simply run. (With the latest Eve Echoes XAPK in the local directory named `eve.xapk`)

If not already happened, clone the repos
```shell
# Replace the url if you want to clone a fork instead
# This command will create a folder eve-echoes-tools
git clone https://github.com/xforce/eve-echoes-tools

# Enter the folder
cd eve-echoes-tools

# Initialize (download) the submodules, if you want to install the other repos manually (because you want
# to install it from a fork, ignore it and look into the section for installing forks
# The section can be found in the INSTALLATION.md file
git submodule update --init --recursive

# Enter the neox-toolx/scripts folder
cd neox-tools/scripts

# Clone unnpk
git clone https://github.com/YJBeetle/unnpk.git

# Go back into the root directory (eve-echoes-tools)
cd ../...
```

> **IMPORTANT**: Make sure that the file `docker/cmd.sh` has Linux line endings (LF) and not Windows line endings (CRLF)
> If you get the error `exec /opt/cmd.sh: no such file or directory` when running the container, you have to fix the line
> endings and re-build the container.

After all repos are downloaded, build the container with this command
```shell
# Inside the eve-echoes-tools directory run
docker build -f docker/Dockerfile -t cookiemagic/evee-tools .
# The dot at the end is crucial!
```

Once the build is completed, run it with this command. Make sure that the xapk in inside the `eve-echoes-tools` directory.
This command will extract all data into the staticdata directory. It will only save the final output, if you want to
use all raw data (to e.g. extract images and other assets), please refer to the manual installation page. The container
will only save the json and script data.

```shell
# When extraction from eve.xapk:
docker run -v$(pwd):/data cookiemagic/evee-tools dump_static /data/eve.xapk /data/staticdata

# When extraction from eve.tar:
docker run -v$(pwd):/data cookiemagic/evee-tools dump_static /data/eve.tar /data/staticdata
```

> On windows adjust the Mount accordingly.
> To run it, you have to put the `$(pwd):/data` part in quotes when using PowerShell:
> ```shell
> # When extraction from eve.xapk:
> docker run -v "$(pwd):/data" cookiemagic/evee-tools dump_static /data/eve.xapk /data/staticdata
> 
> # When extraction from eve.tar:
> docker run -v "$(pwd):/data" cookiemagic/evee-tools dump_static /data/eve.tar /data/staticdata
> ``` 


Now you should have all the static data for Eve Echoes ready to use in `staticdata`. Enjoy.

<br />

## [FSD2JSON](fsd2json)

A simple tool to convert the static data fsd files to json, making it easy to inspect and play with.

## [NeoX Tools](https://github.com/xforce/neox-tools)

Simple tools to interact with the engine files of NeoX as used in Eve Echoes and other NetEase games.

## Installing

> This section does probably also only partially work

All you have to do to build it is clone it and run on of the following:

```
cargo install --path {path to tool}
```

> {path to tool} to be replaced by one of the directories in this repo.


## Getting the APK
The tool does support both extracting from an (x)apk and from a .tar archive. The xapk is required to obtain the full 
data. The easiest way to obtain the most recent data is to back them up from an android device/emulator.

### Prerequisites
An Android device with USB debugging enabled ([link](https://developer.android.com/studio/debug/dev-options)) or an
emulator with ADB (Android Debug Bridge) enabled. If you use Bluestacks, start the instance and enable ADB in the
"Advanced" section of the settings.

You will need these tools:
- [Android Platform Tools](https://developer.android.com/tools/releases/platform-tools#downloads)
- [Android Backup Extractor](https://github.com/nelenkov/android-backup-extractor)
- Java Runtime Environment

### Create Backup
Connect you device to ADB, when using an emulator look up the Host/Port in the settings and run e.g.
```shell
adb connect 127.0.0.1:63000
```
After you have connected your device, you can start the backup. You will have to confirm the backup on your device.
```shell
adb backup -f backup.ab -apk -obb com.netease.eve.en
```

This will create a backup named `backup.ab`. Take this file and run the `.jar` from android-backup-extractor:
```shell
# Both the abe.jar and the backup.ab have to be in the same directory
java -jar abe.jar unpack backup.ab eve.tar
```
The generated `.tar` can be used for this tool to extract data from instead of an `.xapk`.


