#!/usr/bin/env python2
import mmh3
import argparse
import json
import tempfile
import zipfile
import subprocess
import os
import shutil
import fnmatch
import importlib
import imp
import inspect
import contextlib

parser = argparse.ArgumentParser(
    description='Dump all the static data out of the Eve Echoes XAPK')
parser.add_argument('apk', type=str, action='store',
                    help="APK File to extract static data from")
parser.add_argument('outdir', type=str, action='store',
                    help="Target directory to extract the static data to")

args = parser.parse_args()


@contextlib.contextmanager
def tempdir():
    dirpath = tempfile.mkdtemp()

    def cleanup():
        shutil.rmtree(dirpath)
    yield dirpath
    cleanup()


def execute(argv, env=os.environ):
    try:
        subprocess.check_call(argv, env=env)
        return 0
    except subprocess.CalledProcessError as e:
        return e.returncode


def execute_stdout(argv, env=os.environ):
    try:
        output = subprocess.check_output(
            argv, stderr=subprocess.STDOUT, env=env)
        return output
    except subprocess.CalledProcessError as e:
        print(e.output)
        raise e

# TODO(alexander): Move this to neox-tools
# In some way at least, maybe strip it down a bit idk


def dump_scripts(apk):
    with tempdir() as apk_temp_dir:
        # Script stuff
        with zipfile.ZipFile(apk) as apk_zip:
            apk_zip.extractall(apk_temp_dir)
            script_npk = os.path.join(apk_temp_dir, "assets", "script.npk")
        with tempdir() as script_extract_dir:
            execute(["cargo", "run", "--release", "--manifest-path=neox-tools/Cargo.toml",
                     "--", "x", script_npk, script_extract_dir])
            for filename in os.listdir(os.path.join(script_extract_dir, "script")):
                if not filename.endswith(".nxs"):
                    continue
                script_redirect_out = execute_stdout(["python2", "neox-tools/scripts/script_redirect.py", os.path.join(script_extract_dir, "script",
                                                                                                                       filename)])
                c_pyc_file = tempfile.NamedTemporaryFile(
                    mode="wb", delete=False)
                c_pyc_file.write(script_redirect_out)
                c_pyc_file.close()
                pyc_script_file = tempfile.NamedTemporaryFile(
                    mode="wb", delete=False, suffix=".pyc")
                pyc_script_file.close()
                execute(["python2", "neox-tools/scripts/pyc_decryptor.py",
                         c_pyc_file.name, pyc_script_file.name])
                os.remove(c_pyc_file.name)
                with tempfile.NamedTemporaryFile(mode="wb", delete=False) as py_file:
                    meow = execute(["python2", "neox-tools/scripts/decompile_pyc.py",
                                    "-o", py_file.name, pyc_script_file.name])
                    os.remove(pyc_script_file.name)
                    py_file.close()
                    if meow == 0:
                        # Yay
                        py_file = open(py_file.name)
                        py_file.seek(0)
                        lines = py_file.readlines()
                        py_file.close()
                        if lines[4].startswith("# Embedded file name:"):
                            filename = lines[4].replace(
                                "# Embedded file name: ", "")
                            filename = filename.replace("\\", "/")
                            filename = filename.replace("\n", "")
                            print(filename)
                            filedir = os.path.join(
                                args.outdir, "script", os.path.dirname(filename))
                            if not os.path.exists(filedir):
                                os.makedirs(filedir)
                            shutil.copy(py_file.name, os.path.join(
                                args.outdir, "script", filename))
                    os.remove(py_file.name)


def dump_static_data_fsd(xapk_temp_dir):
    obb_path = os.path.join(xapk_temp_dir, "Android",
                            "obb", "com.netease.eve.en")
    for filename in os.listdir(obb_path):
        with zipfile.ZipFile(os.path.join(obb_path, filename), 'r') as obb_zip:
            obb_zip.extractall(obb_path)
    static_data_dir = os.path.join(
        xapk_temp_dir, "Android", "obb", "com.netease.eve.en", "res", "staticdata")
    static_data_dir = os.path.abspath(os.path.realpath(static_data_dir))
    for root, dirnames, filenames in os.walk(static_data_dir):
        for filename in fnmatch.filter(filenames, '*.sd'):
            dir = os.path.relpath(root, static_data_dir)
            print(dir)
            # print(os.path.dirname(filename))
            sd_json_dir = os.path.join(args.outdir, "staticdata", dir)
            if not os.path.exists(sd_json_dir):
                os.makedirs(sd_json_dir)
            execute(["cargo", "run", "--release", "--bin",
                     "fsd2json", "--", "-o", sd_json_dir, os.path.join(root, filename)])


def transform_dict(d):
    for k, v in d.items():
        if type(v) is dict:
            transform_dict(v)
        elif callable(v):
            d[k] = "".join(str(inspect.getsourcelines(v)[0]).strip(
                "['\\n']").split("': ")[1:])


def convert_files(root_dir, sub):
    root_dir = os.path.abspath(os.path.realpath(root_dir))
    for root, dirnames, filenames in os.walk(root_dir):
        for filename in fnmatch.filter(filenames, '*.py'):
            directory = os.path.relpath(root, root_dir)
            filename = os.path.join(directory, filename)
            filename = os.path.join(root_dir, filename)
            if os.path.isfile(filename):
                try:
                    mod = imp.load_source(filename.replace(
                        "/", ".").replace(".py", ""), filename)
                    # mod = importlib.import_module(
                    #     filename.replace("/", ".").replace(".py", ""))
                    members = dir(mod)

                    members = [m for m in members if not m.startswith(
                        "__") and not m == "_reload_all"]
                    is_just_data = False
                    if len(members) == 1:
                        is_just_data = members[0] == "data"
                        #  Only 1 export

                    if len(members) > 0:
                        out_file = os.path.join(
                            args.outdir, "py_data", sub, directory, os.path.basename(filename).replace(".py", ".json"))

                        out_dir = os.path.dirname(out_file)
                        if not os.path.exists(out_dir):
                            os.makedirs(out_dir)
                        with open(out_file, "w") as f:
                            if is_just_data:
                                if type(mod.data) is dict:
                                    transform_dict(mod.data)
                                json.dump(mod.data, f, ensure_ascii=False)

                        # TODO(alexander): For now we don't care
                        # Will have more probably in the future :)
                        if not is_just_data:
                            os.remove(out_file)

                except:
                    print("Failed to convert %s" % filename)
                    pass


with tempdir() as xapk_temp_dir:
    with zipfile.ZipFile(args.apk, 'r') as zip_ref:
        zip_ref.extractall(xapk_temp_dir)

        apk = os.path.join(xapk_temp_dir, "com.netease.eve.en.apk")
        dump_scripts(apk)

        # Static Data stuff
        dump_static_data_fsd(xapk_temp_dir)

        # Script data to json
        convert_files(os.path.join(args.outdir, "script", "data"), "data")
        convert_files(os.path.join(args.outdir, "script",
                                   "data_common"), "data_common")

# print(script_npk)
