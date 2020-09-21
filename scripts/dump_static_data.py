#!/usr/bin/env python
# -*- coding: utf-8 -*-
import mmh3
import argparse
import json
import tempfile
import zipfile
import subprocess
import os
import shutil
import fnmatch
import inspect
import contextlib
import parso
import sys
from multiprocessing import Pool, Lock

PYTHON3 = sys.version_info >= (3, 0)


def which(program):
    import os

    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None


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


def init(l):
    global lock
    lock = l


def dump_script(filename, script_extract_dir):
    if not filename.endswith(".nxs"):
        return
    script_redirect_out = execute_stdout([sys.executable, "neox-tools/scripts/script_redirect.py", os.path.join(script_extract_dir,
                                                                                                                filename)])
    c_pyc_file = tempfile.NamedTemporaryFile(
        mode="wb", delete=False)
    c_pyc_file.write(script_redirect_out)
    c_pyc_file.close()
    pyc_script_file = tempfile.NamedTemporaryFile(
        mode="wb", delete=False, suffix=".pyc")
    pyc_script_file.close()
    execute([sys.executable, "neox-tools/scripts/pyc_decryptor.py",
             c_pyc_file.name, pyc_script_file.name])
    os.remove(c_pyc_file.name)
    with tempfile.NamedTemporaryFile(mode="wb", delete=False) as py_file:
        execute([sys.executable, "neox-tools/scripts/decompile_pyc.py",
                 "-o", py_file.name, pyc_script_file.name])
        os.remove(pyc_script_file.name)
        py_file.close()
        py_file = open(py_file.name)
        py_file.seek(0)
        lines = py_file.readlines()
        py_file.close()
        # TODO(alexander): Argh
        if len(lines) > 5 and lines[5].startswith("# Embedded file name:"):
            print(filename, pyc_script_file.name)
            filename = lines[5].replace(
                "# Embedded file name: ", "")
            filename = filename.replace("\\", "/")
            filename = filename.replace("\n", "")
            filedir = os.path.join(
                args.outdir, "script", os.path.dirname(filename))
            try:
                lock.acquire()
                if not os.path.exists(filedir):
                    os.makedirs(filedir)
            finally:
                lock.release()

            shutil.copy(py_file.name, os.path.join(
                args.outdir, "script", filename))
        else:
            # TODO(alexander): Move to scripts_failed dir?
            pass
        os.remove(py_file.name)


def dump_script_unpack(args):
    return dump_script(*args)


def dump_scripts(apk):
    with tempdir() as apk_temp_dir:
        # Script stuff
        with zipfile.ZipFile(apk) as apk_zip:
            apk_zip.extractall(apk_temp_dir)
            script_npk = os.path.join(apk_temp_dir, "assets", "script.npk")
        with tempdir() as script_extract_dir:
            if which("npktool") is not None:
                execute(["npktool", "x", '-d', script_extract_dir, script_npk])
            else:
                execute(["cargo", "run", "--release", "--manifest-path=neox-tools/Cargo.toml",
                         "--", "x", '-d', script_extract_dir, script_npk])
            lock = Lock()
            import multiprocessing
            pool = Pool(int(multiprocessing.cpu_count()),
                        initializer=init, initargs=(lock,))
            files = []
            for root, dirnames, filenames in os.walk(script_extract_dir):
                for filename in filenames:
                    files.append((filename, root))
            import random
            random.shuffle(files)
            pool.map_async(dump_script_unpack, files).get(9999999)


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
            if which("fsd2json") is not None:
                execute(["fsd2json", "-o", sd_json_dir,
                         os.path.join(root, filename)])
            else:
                execute(["cargo", "run", "--release", "--bin",
                         "fsd2json", "--", "-o", sd_json_dir, os.path.join(root, filename)])


def transform_node(d):
    import parso
    if not hasattr(d, 'children'):
        return d
    for k, v in enumerate(d.children):
        if type(v) is parso.python.tree.Lambda:
            d.children[k] = parso.python.tree.String(
                "'%s'" % v.get_code().replace('\n', '').replace('  ', ''), start_pos=(0, 0))
            pass
        else:
            transform_node(v)
    return d


def extract_data_from_python(filename, directory, sub):
    if os.path.isfile(filename):
        import ast
        try:
            # NOTE(alexander): Ideally we wouldn't use 3.9 here, but parso doesn't support
            # 2.7, and since it has code to handle errors quite well, parsing will finish regardless
            # in most cases
            module = parso.parse(
                open(filename).read(), version="3.9")
            # NOTE(alexander): This expects ExprStmt (which is an assignemnt), to be in the root level of the file

            # Define our target data thing
            out_data = {}

            # Extract all top-level constant assignments
            if filename.endswith("reprocess.py"):
                filename = filename

            for expr in module.children:
                if type(expr) is not parso.python.tree.ExprStmt and expr.type != 'simple_stmt':
                    continue

                # Fixup some stuff
                # TODO(alexander): We should probably do a recursive walk down here
                # Instead of these kind of hacks
                # Look for exprstmt with a name inside
                # And use that if we find id
                if expr.type == 'simple_stmt':
                    # This 'ususally' works for extracting the contained ExprStmt
                    # HACK HACK HACK
                    expr = expr.children[0]

                # Make sure this is actually an ExprStmt now
                if type(expr) is not parso.python.tree.ExprStmt:
                    continue

                # This is not something we can handle yet
                # It looks like the first child is not a 'Name'
                if not hasattr(expr.children[0], 'value'):
                    continue

                # Get the name of the variable
                # NOTE(alexander): This assumes that the first child of a ExprStmt is a Name node
                # Is that really always the case?
                name = expr.children[0].value
                if name != '_reload_all' and not name.startswith('#'):
                    # Ignore all errors here, they are either parse error
                    # Or literal_eval error, where python can't load a dict, because some things are missing
                    # TODO(alexander): Need special handling for certain things
                    # like `_GOOD_TIPS3 = _t('购买')` where we have to somehow remove _t function from the ParseTree
                    try:
                        # Remove things like lambda from our object
                        cleaned_node = transform_node(
                            expr.children[2])

                        # Generate the corresponding Python Code for this Node
                        cleaned_code = cleaned_node.get_code(
                            include_prefix=False).strip()

                        # Parse the resulting code into a python 'object'
                        data = ast.literal_eval(cleaned_code)

                        # Put the data in our output object
                        if type(data) is not dict or len(data.keys()) > 0:
                            out_data[name] = data

                    except:
                        pass

            if len(out_data.keys()) == 0:
                return
            import io
            out_file = os.path.join(args.outdir, "py_data", sub, directory, os.path.basename(
                filename).replace(".py", ".json"))

            out_dir = os.path.dirname(out_file)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            # TODO(alexander): If we only have a single data 'child' flatten?
            with io.open(out_file, "w", encoding="utf-8") as f:
                j = json.dumps(
                    out_data, ensure_ascii=False, indent=4, encoding='utf8')
                if not PYTHON3:
                    f.write(unicode(j))
                else:
                    f.write(j)

        except (NameError, SyntaxError, SystemError, ImportError, RuntimeError) as e:
            print(e)
            print("Failed to convert %s" % filename)
            raise


def extract_data_from_python_unpack(args):
    return extract_data_from_python(*args)


def convert_files(root_dir, sub):
    root_dir = os.path.abspath(os.path.realpath(root_dir))
    files = []
    for root, dirnames, filenames in os.walk(root_dir):
        for filename in fnmatch.filter(filenames, '*.py'):
            directory = os.path.relpath(root, root_dir)
            filename = os.path.join(directory, filename)
            filename = os.path.join(root_dir, filename)
            files.append((filename, directory, sub))

    pool = Pool()
    pool.map_async(extract_data_from_python_unpack, files).get(9999999)


with tempdir() as xapk_temp_dir:
    with zipfile.ZipFile(args.apk, 'r') as zip_ref:
        zip_ref.extractall(xapk_temp_dir)

        for filename in os.listdir(xapk_temp_dir):
            if filename.endswith(".apk"):
                apk = os.path.join(xapk_temp_dir, filename)
                dump_scripts(apk)

        # Static Data stuff
        dump_static_data_fsd(xapk_temp_dir)

        # Script data to json
        convert_files(os.path.join(args.outdir, "script", "data"), "data")
        convert_files(os.path.join(args.outdir, "script",
                                   "data_common"), "data_common")

# print(script_npk)
