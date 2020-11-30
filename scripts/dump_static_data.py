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
import zlib
import collections
from multiprocessing import Pool, Lock

PYTHON3 = sys.version_info >= (3, 0)


class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)


def error(msg):
    prefix = '\033[1m\033[31mERROR\033[0m' if os.isatty(1) else 'ERROR'
    print('%s: %s' % (prefix, msg))
    sys.exit(1)


def warn(msg):
    warn.warned = True
    prefix = '\033[1m\033[93mWARNING\033[0m' if os.isatty(1) else 'WARNING'
    print('%s: %s' % (prefix, msg))


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
parser.add_argument('--patch', type=str, action='store',
                    help="Patch directory to use")

args = parser.parse_args()


@contextlib.contextmanager
def tempdir():
    dirpath = tempfile.mkdtemp()

    def cleanup():
        shutil.rmtree(dirpath)
    try:
        yield dirpath
    except Exception as e:
        raise e
    finally:
        cleanup()


def execute(argv, env=os.environ):
    try:
        subprocess.check_call(argv, env=env)
        return 0
    except subprocess.CalledProcessError as e:
        return e.returncode


def execute_stdout(argv, no_output=False, env=os.environ):
    try:
        output = subprocess.check_output(
            argv, stderr=subprocess.STDOUT, env=env)
        return output
    except subprocess.CalledProcessError as e:
        if not no_output:
            print(e.output)
        raise e

# START of Patch file management


patch_file_list = collections.OrderedDict()


def process_patch_files(patch_file_dir):
    with open(os.path.join(patch_file_dir, "0", "1", "2081783950193513057"), "rb") as f:
        compressed_filelist = f.read()
    filelist = zlib.decompress(compressed_filelist)
    if type(filelist) is not str:
        filelist = filelist.decode('utf-8')

    m = collections.OrderedDict()
    for line in filelist.splitlines():
        info = line.split('\t')
        patch_file = check_patch_file_exists(info[1])
        filename = str(info[5])

        if patch_file is not None:
            m[filename] = patch_file

    global patch_file_list
    patch_file_list = m


def check_patch_file_exists(patch_file):
    if patch_file is not None:
        f0 = os.path.join(args.patch, "0", "0", str(patch_file))
        f1 = os.path.join(args.patch, "0", "1", str(patch_file))
        if os.path.exists(f0):
            return f0
        elif os.path.exists(f1):
            return f1
    return None


def patch_file_for_path(path):
    print("Looking up patch for file {}".format(path))
    if path in patch_file_list:
        patch_file = patch_file_list[path]
        print("Patch file found for {} at {}".format(path, patch_file))
        return patch_file

    print("No patch for file {}".format(path))
    return None

# END of Patch file management

# TODO(alexander): Move this to neox-tools
# In some way at least, maybe strip it down a bit idk


def init(l):
    global lock
    lock = l


def dump_script(filename, script_extract_dir, is_patch=False):
    if not filename.endswith(".nxs") and is_patch == False:
        return True

    try:
        file_path = ""
        if is_patch:
            file_path = filename
        else:
            file_path = os.path.join(script_extract_dir, filename)

        script_redirect_out = execute_stdout(
            [sys.executable, "neox-tools/scripts/script_redirect.py", file_path], True)
    except subprocess.CalledProcessError as e:
        if e.returncode >= 132:
            return False
        else:
            return True

    c_pyc_file = tempfile.NamedTemporaryFile(
        mode="wb", delete=False)
    c_pyc_file.write(script_redirect_out)
    c_pyc_file.close()
    pyc_script_file = tempfile.NamedTemporaryFile(
        mode="wb", delete=False, suffix=".pyc")
    pyc_script_file.close()
    if execute([sys.executable, "neox-tools/scripts/pyc_decryptor.py",
                c_pyc_file.name, pyc_script_file.name]) != 0:
        from shutil import copyfile
        filedir = os.path.join(
            args.outdir, "failed", os.path.dirname(filename))
        try:
            lock.acquire()
            if not os.path.exists(filedir):
                os.makedirs(filedir)
        finally:
            lock.release()
        copyfile(c_pyc_file.name, os.path.join(
            args.outdir, "failed", filename))

    os.remove(c_pyc_file.name)
    with tempfile.NamedTemporaryFile(mode="wb", delete=False) as py_file:
        if execute([sys.executable, "neox-tools/scripts/decompile_pyc.py",
                    "-o", py_file.name, pyc_script_file.name]) != 0:
            from shutil import copyfile
            filedir = os.path.join(
                args.outdir, "failed", os.path.dirname(filename))
            try:
                lock.acquire()
                if not os.path.exists(filedir):
                    os.makedirs(filedir)
            finally:
                lock.release()
            copyfile(pyc_script_file.name, os.path.join(
                args.outdir, "failed", filename + ".pyc"))
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

            nxs_filename = os.path.splitext(filename)[0] + ".nxs"
            patch_filename = patch_file_for_path(nxs_filename)

            if is_patch == False and patch_filename is not None:
                print("Ignoring file in favor of patch, processing...")
                dump_script(patch_filename, script_extract_dir, True)
            else:
                print("Writing final script to {}".format(filename))
                shutil.copy(py_file.name, os.path.join(
                    args.outdir, "script", filename))
        else:
            # TODO(alexander): Move to scripts_failed dir?
            pass
        os.remove(py_file.name)

    return True


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

            # Prepare data for parallel execution
            files = []
            for root, dirnames, filenames in os.walk(script_extract_dir):
                for filename in filenames:
                    files.append((filename, root))

            # Attempt to distribute larger files over more executors
            # in testing this does slightly improve things
            import random
            random.shuffle(files)

            # Create pool for parallel execution
            # The lock here is used to synchronize directory create calls
            lock = Lock()
            import multiprocessing
            pool = Pool(int(multiprocessing.cpu_count()),
                        initializer=init, initargs=(lock,))

            if len(files) > 0:
                # Make sure we even have a compatible decrypt plugin available
                # If we don't, just abort and tell the user such, nothing else we can do.
                file = files[0]
                init(lock)
                if not dump_script_unpack(file):
                    warn(
                        "Script redirect decrypt plugin not found, disable script decompilation and script data extraction")
                    return

                files = files[1:]
                pool.map_async(dump_script_unpack, files).get(9999999)


def dump_static_data_fsd(xapk_temp_dir):
    obb_path = os.path.join(xapk_temp_dir, "Android",
                            "obb", "com.netease.eve.en")
    for filename in os.listdir(obb_path):
        with zipfile.ZipFile(os.path.join(obb_path, filename), 'r') as obb_zip:
            obb_zip.extractall(obb_path)

    # Path to staticdata inside xapk
    static_data_dir = os.path.join(
        xapk_temp_dir, "Android", "obb", "com.netease.eve.en", "res", "staticdata")
    static_data_dir = os.path.abspath(os.path.realpath(static_data_dir))

    for root, dirnames, filenames in os.walk(static_data_dir):
        for filename in fnmatch.filter(filenames, '*.sd'):
            dir = os.path.relpath(root, static_data_dir)
            sd_json_dir = os.path.join(args.outdir, "staticdata", dir)
            if not os.path.exists(sd_json_dir):
                os.makedirs(sd_json_dir)

            sd_file_name = os.path.join(root, filename)
            if args.patch is not None:
                print("Looking for patch for file {}".format(filename))
                hash_source = os.path.join("staticdata", dir, filename)

                patch_file = patch_file_for_path(hash_source)

                if patch_file is not None:
                    print("Copying patch file {} to {}".format(
                        patch_file, sd_file_name))
                    # Copy the patched file over the original :)
                    shutil.copy(patch_file, sd_file_name)

            if which("fsd2json") is not None:
                execute(["fsd2json", "-o", sd_json_dir, sd_file_name])
            else:
                execute(["cargo", "run", "--bin",
                         "fsd2json", "--", "-o", sd_json_dir, sd_file_name])


def transform_node(d):
    """
    This is doing some hack to convert things like lambdas in the dict to a JSON compatible representation.

    For now just doing convert lambda code to string
    """
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

            # Generate output path for extracted python data
            import io
            out_file = os.path.join(args.outdir, "py_data", sub, directory, os.path.basename(
                filename).replace(".py", ".json"))

            out_dir = os.path.dirname(out_file)
            try:
                lock.acquire()
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
            finally:
                lock.release()

            # TODO(alexander): If we only have a single data 'child' flatten?
            with io.open(out_file, "w", encoding="utf-8") as f:
                if not PYTHON3:
                    j = json.dumps(
                        out_data, ensure_ascii=False, indent=4, encoding='utf8')
                    f.write(unicode(j))
                else:
                    j = json.dumps(
                        out_data, ensure_ascii=False, indent=4, cls=SetEncoder)
                    f.write(j)

        except (NameError, SyntaxError, SystemError, ImportError, RuntimeError):
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

    lock = Lock()
    import multiprocessing
    pool = Pool(int(multiprocessing.cpu_count()),
                initializer=init, initargs=(lock,))
    init(lock)
    pool.map_async(extract_data_from_python_unpack, files).get(9999999)


with tempdir() as xapk_temp_dir:

    if args.patch is not None:
        process_patch_files(args.patch)

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
