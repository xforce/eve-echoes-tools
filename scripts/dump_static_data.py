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
PATCH_FILE_INDEX="2081783950193513057"


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
parser.add_argument('--xapk', type=str, action='store',
                    help="XAPK File to extract static data from")
parser.add_argument('outdir', type=str, action='store',
                    help="Target directory to extract the static data to")

parser.add_argument('-u', '--unpackdir', type=str, action='store',
                    help="Target directory to unpack files into")

parser.add_argument('-g', '--gamedatadir', type=str, action='store',
                    help="Target directory to reconstruct the game data structure")

parser.add_argument('-p', '--patch', type=str, action='store',
                    help="Patch directory to use")

args = parser.parse_args()

def yes_or_no(question):
    reply = str(raw_input(question+' (y/n): ')).lower().strip()
    if reply[0] == 'y':
        return True
    if reply[0] == 'n':
        return False
    else:
        return yes_or_no("Uhhhh... please enter ")

@contextlib.contextmanager
def tempdir_if_required(dirpath):

    if dirpath is None:
        cleanup_needed = True
        dirpath = tempfile.mkdtemp() 
    else:
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        # else:
        #     if yes_or_no('Clear folder?: {}'.format(dirpath)):
        #         shutil.rmtree(dirpath)
        cleanup_needed = False

    def cleanup(cleanup_needed):
        if cleanup_needed:
            shutil.rmtree(dirpath)
    try:
        yield dirpath
    except Exception as e:
        raise e
    finally:
        cleanup(cleanup_needed)


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

####
# Patch file management
####

patch_file_map = collections.OrderedDict()
patch_file_list = None

def process_patch_file_listing(patch_file_dir):
    with open(os.path.join(patch_file_dir, "0", "1", PATCH_FILE_INDEX), "rb") as f:
        compressed_filelist = f.read()
    filelist = zlib.decompress(compressed_filelist)
    if type(filelist) is not str:
        filelist = filelist.decode('utf-8')

    global patch_file_list 
    patch_file_list = os.path.join(patch_file_dir, "filelist.txt")
    with open(patch_file_list, "w") as f:
        f.write(filelist)

    m = collections.OrderedDict()
    lines = filelist.splitlines()
    numLines = len(lines)
    print("Parsing", numLines, "patch files...", end='')
    i = 0
    for line in lines:
        i += 1
        print('\r', end='') 
        print('Parsing patch entry:', i, "/", numLines, end='')
        info = line.split('\t')
        patch_file = str(info[1])
        filename = str(info[5])
        m[patch_file] = filename

    global patch_file_map
    patch_file_map = m
    print() # To run next print on a new line


def apply_patch_files(patch_file_dir, game_data_dir):
    for root, dirnames, filenames in os.walk(patch_file_dir):
        for filename in filenames:
            if filename in patch_file_map:
                patch_dest_file_name = patch_file_map[filename]
                patch_file_path_src = os.path.join(root, filename)
                patch_file_path_dest = os.path.join(game_data_dir, patch_dest_file_name)
                patch_file_path_dest_dir = os.path.dirname(patch_file_path_dest)

                if not os.path.exists(patch_file_path_dest_dir):
                    os.makedirs(patch_file_path_dest_dir)
                
                shutil.copyfile(patch_file_path_src, patch_file_path_dest)
                #os.rename(os.path.join(patch_file_path_dest_dir, filename), patch_file_path_dest)
            else:
                warn('Patch file not found in index! {}'.format(filename))

# END of Patch file management

# TODO(alexander): Move this to neox-tools
# In some way at least, maybe strip it down a bit idk

def init(l, pfl):
    global lock
    global patch_file_map
    lock = l
    patch_file_map = pfl

####
# Parse scripts
####

def dump_script(filename, script_extract_dir):
    if not filename.endswith(".nxs"):
        return True

    try:
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
            shutil.copy(py_file.name, os.path.join(args.outdir, "script", filename))
        else:
            # TODO(alexander): Move to scripts_failed dir?
            pass
        os.remove(py_file.name)

    return True


def dump_script_unpack(args):
    return dump_script(*args)

####
# Unpack Android files
####

def dump_scripts(apk):
    with tempdir() as apk_temp_dir:
        # Script stuff
        with zipfile.ZipFile(apk) as apk_zip:
            apk_zip.extractall(apk_temp_dir)
            dump_scripts_from_apk_data(apk_temp_dir)

def dump_static_data_fsd(xapk_temp_dir):
    obb_path = os.path.join(xapk_temp_dir, "Android",
                            "obb", "com.netease.eve.en")
    for filename in os.listdir(obb_path):
        with zipfile.ZipFile(os.path.join(obb_path, filename), 'r') as obb_zip:
            obb_zip.extractall(obb_path)

    # Path to staticdata inside xapk
    static_data_dir = os.path.join(
        xapk_temp_dir, "Android", "obb", "com.netease.eve.en")

    dump_static_data_fsd_from_obb_data(static_data_dir)

####
# Unpack Game Data
####

def dump_from_unpacked_data(unpacked_dir, output_dir):
    assets_dir = os.path.join(unpacked_dir, "assets")

    npk_files = []
    for root, dirnames, filenames in os.walk(assets_dir):
        for filename in fnmatch.filter(filenames, '*.npk'):
            npk_file_name = os.path.join(root, filename)
            npk_files.append(npk_file_name)

    if which("npktool") is not None:
        execute_cmds = ["npktool"]
    else:
        execute_cmds = ["cargo", "run", "--release", "--manifest-path=neox-tools/Cargo.toml", "--"]

    execute_cmds.extend(['x', '-d', output_dir])

    global patch_file_list
    if patch_file_list is not None:
        execute_cmds.extend(['-f', patch_file_list])

    execute_cmds.extend(npk_files)
    execute(execute_cmds)

def search_for_scripts(game_data_dir):
    # Prepare data for parallel execution
    files = []
    for root, dirnames, filenames in os.walk(game_data_dir):
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
                initializer=init, initargs=(lock,patch_file_map,))

    if len(files) > 0:
        # Make sure we even have a compatible decrypt plugin available
        # If we don't, just abort and tell the user such, nothing else we can do.
        file = files[0]
        init(lock,patch_file_map)
        if not dump_script_unpack(file):
            warn(
                "Script redirect decrypt plugin not found, disable script decompilation and script data extraction")
            return

        files = files[1:]
        pool.map_async(dump_script_unpack, files).get(9999999)

def process_static_data(game_data_dir):
    for root, dirnames, filenames in os.walk(game_data_dir):
        for filename in fnmatch.filter(filenames, '*.sd'):
            dir = os.path.relpath(root, game_data_dir)
            sd_json_dir = os.path.join(args.outdir, dir)
            if not os.path.exists(sd_json_dir):
                os.makedirs(sd_json_dir)

            sd_file_name = os.path.join(root, filename)

            if which("fsd2json") is not None:
                execute(["fsd2json", "-o", sd_json_dir, sd_file_name])
            else:
                execute(["cargo", "run", "--bin",
                         "fsd2json", "--", "-o", sd_json_dir, sd_file_name])

####
# Transform Python Data
####

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
                initializer=init, initargs=(lock,patch_file_map,))
    init(lock,patch_file_map,)
    pool.map_async(extract_data_from_python_unpack, files).get(9999999)

####
# Main
####

if __name__ == '__main__':

    if args.unpackdir is None and args.xapk is None:
        print('You must give either an unpacked data directory, or an (X)APK containing all the assets.')
    else:
        with tempdir_if_required(args.unpackdir) as unpack_dir:
            with tempdir_if_required(args.gamedatadir) as game_data_dir:
                print('----------------------------')
                print('Unpack Data:', unpack_dir)
                print('Game Data:', game_data_dir)
                print('Output Folder:', args.outdir)
                print('----------------------------')

                ## Unpack XAPK if required
                if args.xapk is not None:
                    with tempdir() as xapk_temp_dir:
                        print('Unpacking XAPK:', xapk_temp_dir)
                        with zipfile.ZipFile(args.xapk, 'r') as zip_ref:
                            zip_ref.extractall(xapk_temp_dir)

                            for root, dirnames, filenames in os.walk(xapk_temp_dir):
                                for filename in filenames:
                                    file_path = os.path.join(root, filename)
                ## Unpack APK to (temp directory)
                                    if filename.endswith(".apk"):
                                        print('Unpacking APK files')
                                        with zipfile.ZipFile(file_path) as apk_zip:
                                            apk_zip.extractall(unpack_dir)
                ## Unpack OBB to (temp directory)\assets
                ## Depending on APK - this might already be in place
                                    elif filename.endswith(".obb"):
                                        print('Unpacking OBB files')
                                        obb_unpack = os.path.join(unpack_dir, 'assets')
                                        with zipfile.ZipFile(file_path) as obb_zip:
                                            obb_zip.extractall(obb_unpack)
                
                ## Parse Patch file listing (if it exists)
                if args.patch is not None:
                    process_patch_file_listing(args.patch)

                ## Extract all NPK files to game_data folder
                #dump_from_unpacked_data(unpack_dir, game_data_dir)

                ## Copy OBB res/* to game_data folder
                print('Moving static data into game data...')
                static_data_src = os.path.join(unpack_dir, 'assets', 'res', 'staticdata')
                static_data_dest = os.path.join(game_data_dir, 'staticdata')
                shutil.copytree(static_data_src, static_data_dest, dirs_exist_ok=True)
            
                ## Copy patchfiles into game_data and rename            
                if args.patch is not None:
                    apply_patch_files(args.patch, game_data_dir)

                ## Process all files in output 
                print('Searching scripts...')
                search_for_scripts(game_data_dir)
                print('Processing static data...')
                process_static_data(game_data_dir)

                ## Convert Python data files
                convert_files(os.path.join(args.outdir, "script", "data"), "data")
                convert_files(os.path.join(args.outdir, "script",
                                            "data_common"), "data_common")