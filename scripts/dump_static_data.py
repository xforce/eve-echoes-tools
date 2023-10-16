#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tarfile
from typing import Dict, Tuple, List

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
PATCH_FILE_INDEX = "2081783950193513057"
NO_SCRIPT = False


def check_xdis():
    # xdis has a hardcoded list with all existing python versions. However, this list is not up-to-date and there are
    # missing versions. Going further, we can't use the latest version of xdis because it has breaking changes.
    # Because the decompilation happens in separate threads, the user will have to fix this manually
    from xdis import magics
    from xdis.op_imports import version_tuple_to_str
    ver_str = version_tuple_to_str(sys.version_info)
    print("Detected python version " + ver_str)
    if ver_str in magics.canonic_python_version:
        return  # xdis knows our version, nothing to do for us
    warn(f"xdis does not know our python version, it has to be inserted manually into {magics.__file__}")
    warn(f"Please insert our version {ver_str} into the correct \"add_canonic_versions\" command")
    warn("You can just select the one with the same major and minor version number")
    warn("For example if we have the version 3.8.18, insert this number into the line with the other 3.8.x versions")
    error(f"Please insert version {ver_str} into {magics.__file__}")


class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)


def error(msg: str):
    prefix = '\033[1m\033[31mERROR\033[0m' if os.isatty(1) else 'ERROR'
    print('%s: %s' % (prefix, msg))
    sys.exit(1)


def warn(msg: str):
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
parser.add_argument('--tar', type=str, action="store",
                    help="TAR File to extract static data from")
parser.add_argument('--auto', action="store_true",
                    help="Detect the file type (xapk/tar) from the file name")
parser.add_argument('outdir', type=str, action='store',
                    help="Target directory to extract the static data to")

parser.add_argument('-u', '--unpackdir', type=str, action='store',
                    help="Target directory to unpack files into")

parser.add_argument('-g', '--gamedatadir', type=str, action='store',
                    help="Target directory to reconstruct the game data structure")

parser.add_argument('-p', '--patch', type=str, action='store',
                    help="Patch directory to use")

parser.add_argument('-patch', '--patch_game_files', action='store_true',
                    help="Only patch saved game files, skip unpacking")

parser.add_argument('-s', '--skip_delete', action='store_true',
                    help="Don't ask if the unpack and gamedata dir should get deleted (those directory wont be deleted)")

parser.add_argument('--no_script', action='store_true',
                    help="Will not try to extract .nxs files and only process .sd files.")

args = parser.parse_args()


def yes_or_no(question: str):
    reply = str(input(question + ' (y/n): ')).lower().strip()
    if reply[0] == 'y':
        return True
    if reply[0] == 'n':
        return False
    else:
        return yes_or_no("Uhhhh... please enter ")


@contextlib.contextmanager
def tempdir_if_required(dirpath, skip_clear_question=False):
    if dirpath is None:
        cleanup_needed = True
        dirpath = tempfile.mkdtemp()
    else:
        if os.path.exists(dirpath) and not skip_clear_question and yes_or_no('Clear folder?: {}'.format(dirpath)):
            shutil.rmtree(dirpath)

        os.makedirs(dirpath, exist_ok=True)
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
    print("Parsing", numLines, "patch files...", end='\r')
    i = 0
    for line in lines:
        i += 1
        print('Parsing patch entry:', i, "/", numLines, end='\r')
        info = line.split('\t')
        patch_file = str(info[1])
        filename = str(info[5])
        m[patch_file] = filename

    global patch_file_map
    patch_file_map = m


def apply_patch_files(patch_file_dir, game_data_dir):
    for root, _, filenames in os.walk(patch_file_dir):
        i = 0
        numFiles = len(filenames)
        for filename in filenames:
            i += 1
            if filename.startswith('.') or filename in ['filelist.txt', PATCH_FILE_INDEX]:
                pass
            elif filename in patch_file_map:
                print('Applying patch file', i, "/", numFiles, filename, end='\r')
                patch_file_path_src = os.path.join(root, filename)
                patch_file_path_dest = os.path.join(game_data_dir, patch_file_map[filename])
                patch_file_path_dest_dir = os.path.dirname(patch_file_path_dest)

                if not os.path.exists(patch_file_path_dest_dir):
                    os.makedirs(patch_file_path_dest_dir, exist_ok=True)

                shutil.copyfile(patch_file_path_src, patch_file_path_dest)
            else:
                warn('Patch file not found in index! {}'.format(filename))
    print('\033[KPatch files applied.')


# TODO(alexander): Move this to neox-tools
# In some way at least, maybe strip it down a bit idk

def init(l):
    global lock
    lock = l


####
# Process Game Data
####

def search_for_scripts(game_data_dir: str):
    # Prepare data for parallel execution
    files = []  # type: List[Tuple[str, str, str]]
    for root, dirnames, filenames in os.walk(game_data_dir):
        for filename in filenames:
            dir = os.path.relpath(root, game_data_dir)
            files.append((filename, dir, root))

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

    if len(files) == 0:
        warn("No files found")
        return
    if not NO_SCRIPT:
        # Make sure we even have a compatible decrypt plugin available
        # If we don't, just abort and tell the user such, nothing else we can do.
        file = next(filter(lambda file_tuple: file_tuple[0].endswith(".nxs"), files), None)

        init(lock, )
        if not parse_file_unpack(file):
            warn("Script redirect decrypt plugin not found, aborted script decompilation and script data extraction")
            return

    files = files[1:]
    pool.map_async(parse_file_unpack, files).get(9999999)


def parse_file(filename, relative_dir, root_dir):
    if filename.endswith(".nxs"):
        if NO_SCRIPT:
            return True
        return dump_script(filename, relative_dir, root_dir)
    elif filename.endswith('.sd'):
        return dump_sd(filename, relative_dir, root_dir)
    else:
        return True


def dump_sd(filename, relative_dir, root_dir):
    file_path = os.path.join(root_dir, filename)
    sd_json_dir = os.path.join(args.outdir, relative_dir)
    if not os.path.exists(sd_json_dir):
        os.makedirs(sd_json_dir, exist_ok=True)
    if which("fsd2json") is not None:
        execute(["fsd2json", "-o", sd_json_dir, file_path])
    else:
        execute(["cargo", "run", "--bin",
                 "fsd2json", "--", "-o", sd_json_dir, file_path])

    return True


def dump_script(filename, relative_dir, root_dir):
    try:
        file_path = os.path.join(root_dir, filename)

        script_redirect_out = execute_stdout(
            [sys.executable, "neox-tools/scripts/script_redirect.py", file_path], False)
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
            args.outdir, "failed", relative_dir, os.path.dirname(filename))
        try:
            lock.acquire()
            if not os.path.exists(filedir):
                os.makedirs(filedir, exist_ok=True)
        finally:
            lock.release()
        copyfile(c_pyc_file.name, os.path.join(
            args.outdir, "failed", relative_dir, filename))

    os.remove(c_pyc_file.name)

    with tempfile.NamedTemporaryFile(mode="wb", delete=False) as py_file:
        if execute([sys.executable, "neox-tools/scripts/decompile_pyc.py",
                    "-o", py_file.name, pyc_script_file.name]) != 0:
            from shutil import copyfile
            filedir = os.path.join(
                args.outdir, "failed", relative_dir, os.path.dirname(filename))
            try:
                lock.acquire()
                if not os.path.exists(filedir):
                    os.makedirs(filedir, exist_ok=True)
            finally:
                lock.release()
            copyfile(pyc_script_file.name, os.path.join(
                args.outdir, "failed", relative_dir, filename + ".pyc"))
        os.remove(pyc_script_file.name)

        py_file.close()
        py_file = open(py_file.name)
        py_file.seek(0)
        lines = py_file.readlines()
        py_file.close()

        filename = filename.replace(".nxs", ".py")
        filedir = os.path.join(
            args.outdir, "script", relative_dir)
        try:
            lock.acquire()
            if not os.path.exists(filedir):
                os.makedirs(filedir, exist_ok=True)
        finally:
            lock.release()

        shutil.copy(py_file.name, os.path.join(filedir, filename))
        os.remove(py_file.name)

    return True


def parse_file_unpack(args):
    return parse_file(*args)


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


def cleanup_dict(data: Dict):
    """
    Cleanups up a data dictionary.

    Decodes (recursively) all bytes (both keys and values) to UTF-8 strings to ensure a
    JSON compatible representation (json.dumps raises an error if supplied with bytes).

    :param data: The dict to clean up
    """
    to_delete = []
    to_add = {}
    for k, v in data.items():
        if type(v) is bytes:
            data[k] = v.decode("utf-8")
            print("Decoded value bytes %s to utf-8", v)
        if type(k) is bytes:
            print("Decoded key bytes %s to utf-8", k)
            to_add[k.decode("utf-8")] = data[k]
            to_delete.append(k)
        if type(v) is dict:
            cleanup_dict(v)
    for k in to_delete:
        data.pop(k)
    for k, v in to_add.items():
        data[k] = v


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
                    # This 'usually' works for extracting the contained ExprStmt
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
            cleanup_dict(out_data)

            # Generate output path for extracted python data
            import io
            out_file = os.path.join(args.outdir, "py_data", sub, directory, os.path.basename(
                filename).replace(".py", ".json"))

            out_dir = os.path.dirname(out_file)
            try:
                lock.acquire()
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir, exist_ok=True)
            finally:
                lock.release()

            # TODO(alexander): If we only have a single data 'child' flatten?
            with io.open(out_file, "w", encoding="utf-8") as f:
                if not PYTHON3:
                    j = json.dumps(
                        out_data, ensure_ascii=False, indent=4, encoding='utf8')
                    # ToDo: is this correct? Showing 'unresolved reference' for me and don't find an import
                    f.write(unicode(j))
                else:
                    j = json.dumps(
                        out_data, ensure_ascii=False, indent=4, cls=SetEncoder)
                    f.write(j)

        except (NameError, SyntaxError, SystemError, ImportError, RuntimeError, TypeError):
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
    init(lock, )
    pool.map_async(extract_data_from_python_unpack, files).get(9999999)


####
# Main
####

if __name__ == '__main__':
    if args.unpackdir is None and args.xapk is None and args.tar is None and args.patch_game_files is not True:
        print('You must give either an unpacked data directory, or an (X)APK/TAR containing all the assets.')
        exit(1)

    if args.no_script or NO_SCRIPT:
        warn("Script extraction is disabled, won't process .nxs files")
        NO_SCRIPT = True
    else:
        check_xdis()

    with tempdir_if_required(args.gamedatadir, args.skip_delete) as game_data_dir:
        with tempdir_if_required(args.unpackdir, args.skip_delete) as unpack_dir:
            print('----------------------------')
            print('Unpack Data:', unpack_dir)
            print('Game Data:', game_data_dir)
            print('Output Folder:', args.outdir)
            print('----------------------------')

            ## Parse Patch file listing (if it exists)
            if args.patch is not None:
                process_patch_file_listing(args.patch)
            if args.auto:
                archive_path = args.xapk or args.tar
                if archive_path.endswith("apk"):
                    archive_class = zipfile.ZipFile
                elif archive_path.endswith(".tar"):
                    archive_class = tarfile.TarFile
                else:
                    print(f"Could not detect archive format for file {archive_path}, using zip")
                    archive_class = zipfile.ZipFile
            else:
                archive_path = args.xapk or args.tar
                archive_class = zipfile.ZipFile if args.xapk else tarfile.TarFile
            ## Unpack XAPK if required
            if archive_path is not None:
                print(f"Using extractor: {archive_class.__name__}")
                if not os.path.exists(archive_path):
                    error(f"Path to XAPK/TAR not found: {archive_path}")
                    exit(1)
                with tempdir() as xapk_temp_dir:
                    print('Unpacking XAPK/TAR into', xapk_temp_dir)
                    with archive_class(archive_path, 'r') as archive_ref:
                        print(f"Extracting from {archive_path}")
                        archive_ref.extractall(xapk_temp_dir)
                        # Walk the files
                        for root, dirnames, filenames in os.walk(xapk_temp_dir):
                            for filename in filenames:
                                file_path = os.path.join(root, filename)
                                obb_unpack = os.path.join(unpack_dir, 'assets')
                                ## Unpack APK to (temp directory)
                                if filename.endswith(".apk"):
                                    print(f'Unpacking APK files from {file_path}')
                                    with zipfile.ZipFile(file_path) as apk_zip:
                                        apk_zip.extractall(unpack_dir)
                                ## Unpack OBB to (temp directory)\assets
                                ## Depending on APK - this might already be in place
                                elif filename.endswith(".obb"):
                                    print(f'Unpacking OBB files from {file_path}')
                                    with zipfile.ZipFile(file_path) as obb_zip:
                                        obb_zip.extractall(obb_unpack)

            ## Extract all NPK files to game_data folder
            if args.patch_game_files is not True:
                print('Extracting game assets')
                dump_from_unpacked_data(unpack_dir, game_data_dir)
                ## Copy OBB res/* to game_data folder
                print('Moving static data into game data...')
                static_folders = ["staticdata", "sigmadata", "manual_staticdata"]
                for folder in static_folders:
                    static_data_src = os.path.join(unpack_dir, 'assets', 'res', folder)
                    static_data_dest = os.path.join(game_data_dir, folder)
                    print(f"Coping from {static_data_src} to {static_data_dest}")
                    shutil.copytree(static_data_src, static_data_dest, dirs_exist_ok=True)

            ## Copy patchfiles into game_data and rename
            if args.patch is not None:
                apply_patch_files(args.patch, game_data_dir)

            ## Process all files in output
            if args.no_script or NO_SCRIPT:
                warn("Script extraction is disabled, won't process .nxs files")
            print('Searching scripts...')
            search_for_scripts(game_data_dir)

            ## Convert Python data files
            print('Converting Python data files...')
            convert_files(os.path.join(args.outdir, "script", "data"), "data")
            convert_files(os.path.join(args.outdir, "script",
                                       "data_common"), "data_common")
