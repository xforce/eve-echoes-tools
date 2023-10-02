# Manual Installation
This guide will cover how to install the script including the python decompiler. This guide is made for `Ubuntu` (
other Linux distributions will probably work similar). If you are running Windows, I recommend to install [WSL](https://learn.microsoft.com/de-de/windows/wsl/install)
as this will make it a lot easier.

This instruction was written and tested on `Ubuntu 22.04` on Windows 10 WSL with `Python 3.10.12`.



Table of contents:
<!-- TOC -->
* [Manual Installation](#manual-installation)
    * [Step 0: Install dependencies](#step-0-install-dependencies)
    * [Step 1: Install repos](#step-1-install-repos)
    * [Step 2: Build unnpk](#step-2-build-unnpk)
    * [Step 3: Run it](#step-3-run-it)
      * [XDIS error](#xdis-error)
  * [Installing forks](#installing-forks)
* [Dependency Versions](#dependency-versions)
  * [APT Packages](#apt-packages)
  * [Python Packages](#python-packages)
<!-- TOC -->


### Step 0: Install dependencies
To compile unnpk, you have to install the following dependencies:
```shell
sudo apt install -y build-essential libmagic-dev zlib1g-dev
```

You will also need python. This repo was tested with `Python 3.10.12`. You can install it with
```shell
sudo add-apt-repository ppa:deadsnakes/ppa
# Press [Enter] to confirm it

# Then install python3.10
sudo apt install -y python3.10 python3.10-venv
```
Other versions will probably also work.


### Step 1: Install repos
Those three repositories have to be installed:
- [eve-echoes-tools](https://github.com/xforce/eve-echoes-tools)
- [neox-tools](https://github.com/xforce/neox-tools)
- [unnpk](https://github.com/YJBeetle/unnpk)

If you want to install some from another source, replace the link with your repo link and read the [Installing forks](#installing-forks)
section for instructions on how to install them.

Clone the repository using the following command
```shell
# Clone the main repo, will install it into a folder called eve-echoes-tools
# Replace the link if you want to clone a fork instead
git clone https://github.com/xforce/eve-echoes-tools

# Set up python environment, you might need to change the 'python3' command depending on you installation
python3 -m venv eve-echoes-tools

# Enter the folder
cd eve-echoes-tools

# Initialize (download) the submodules, if you want to install the other repos manually (because you want
# to install it from a fork, ignore it and look into the section for installing forks
git submodule update --init --recursive

# Activate the python venv
source bin/activate

# Install required python packages to you python environment (make sure that you did activate it)
pip install -r requirements.txt
```

### Step 2: Build [unnpk](https://github.com/YJBeetle/unnpk)
```shell
# Deactivate the python environment for now
deactivate

# Install the dev dependencies for unnpk
sudo apt install -y build-essential libmagic-dev zlib1g-dev

# Enter the neox-toolx/scripts folder
cd neox-tools/scripts/unnpk

# Build the repository
# You will get an warning but that can be ignored
make

# Go back to the root directory (eve-echoes-tools)
cd ../../..
```

### Step 3: Run it
```shell
# If not already happened, activate it again
source bin/activate

# Extract data to the staticdata directory
python scripts/dump_static_data.py staticdata --xapk eve.xapk -g staticdata/gamedata
```
After you have run and unpacked

> Note: When decompiling python files, there will be a lot of spam in the console because of failing files.
> You can just ignore it.

> If you also want to use the raw unpacked data, you can add `-u staticdata/unpack` to the command, this will also save
> the unpacked data from the apk which can be processed further.

#### XDIS error
The program might crash with an error similar to this one (you will maybe have another file path and version number):
```
ERROR: Please insert version 3.8.18 into /**/**/eve-echoes-tools/lib/python3.8/site-packages/xdis/magics.py
```
In this case you have to open the specified file and locate the line with the matching major and minor python version
(should be around line 300) starting with the function `add_canonic_versions`.
The major and minor version are the first two numbers. So for example if we have the version `3.8.18` we have to find
the line that contains similar versions, e.g. `3.8`, `3.8.5`. In our case, the correct line looks like this:
```python
add_canonic_versions(
    "3.8b4 3.8.0candidate1 3.8 3.8.0 3.8.1 3.8.2 3.8.3 3.8.4 3.8.5 3.8.6 3.8.7 3.8.8 3.8.9 3.8.10 3.8.11 3.8.12",
    "3.8.0rc1+",
)
```
Depending on your version, this might also be multiple lines. To fix the error, you have to insert your python version
into the first string (separated with a space), in this case it should look like this:
```python
add_canonic_versions(
    # Edit this first string
    "3.8b4 3.8.0candidate1 3.8 3.8.0 3.8.1 3.8.2 3.8.3 3.8.4 3.8.5 3.8.6 3.8.7 3.8.8 3.8.9 3.8.10 3.8.11 3.8.12 3.8.18",
    # Don't touch this second string
    "3.8.0rc1+",
)
```
After that, the error should be gone.


## Installing forks
If you want to install the repos from another source, you have to skip the `git submodule` command and install
them manually:
```shell
# Inside the eve-echoes-tools folder

# Install neox-tools
# Replace the link with the fork link (or leave it)
git clone https://github.com/xforce/neox-tools

cd neox-tools/scripts

# Install unnpk
# Replace the link with the fork link (or leave it)
git clone https://github.com/YJBeetle/unnpk

# Go back into the eve-echoes-tools directory and continue the installation instruction
cd ../..
```

# Dependency Versions
This repo is not actively maintained, in case you want to make use of it in the future, you can find the latest tested
dependency versions here. If the latest versions do not work because they introduced breaking changes, you can revert to
these versions instead.
## APT Packages

| package         | version                  |
|-----------------|--------------------------|
| python3.10      | 3.10.12-1~22.04.2        |
| build-essential | 12.9ubuntu3              |
| libmagic-dev    | 1:5.41-3ubuntu0.1        |
| zlib1g-dev      | 1:1.2.11.dsfg-2ubuntu9.2 |

## Python Packages
You can find these packages in the file `requirements-lock.txt`

| package      | version |
|--------------|---------|
| uncompyle6   | 3.9.0   |
| xdis         | 6.0.5   |
| mmh3         | 4.0.1   |
| spark-parser | 1.8.9   |
| parso        | 0.8.3   |
| click        | 8.1.7   |
| six          | 1.16.0  |
