import os.path
import re
import sys

# xdis.magics contains a list of hardcoded python version strings. However, this file may not contain our current
# python version, so we have to check it and add it if necessary. The magics.py file contains a long list of function
# calls similar to this one:
#   add_canonic_versions(
#       "3.10 3.10.0 3.10.1 3.10.2 3.10.3 3.10.4 3.10.5 3.10.6 3.10.7 3.10.8 3.10.9", "3.10.0rc2"
#   )
# These map the python versions (in this case a selection of 3.10.x versions) onto a base version (in this case 3.10.0rc2)
# We will add our own line that registers our version. Currently only python 3.10.x is supported by this script.
if __name__ == "__main__":
    print("Checking xdis")
    from xdis import magics
    from xdis.op_imports import version_tuple_to_str

    ver_str = version_tuple_to_str(sys.version_info)
    if ver_str in magics.canonic_python_version:
        # xdis knows our python version
        print(f"XDIS knows our version ({ver_str}) already")
        exit(0)
    # Check if we are running python 3.10.xx
    if not re.match(r"3\.10\.\d+", ver_str):
        # Unsupported python version, can't apply auto-patch
        print(f"Unsupported python version {ver_str}, only python3.10.xx is supported")
        exit(135)
    # Python 3.10 detected, applying patch
    file_path = magics.__file__
    if not os.path.exists(file_path):
        print("Error: File not found: " + file_path)
        exit(136)
    # Open xdis/magics.py file in append mode and add our version to the end
    with open(file_path, "a") as file:
        file.write(f"\nadd_canonic_versions('{ver_str}', '3.10.0rc2')\n")
        print(f"Info: Applied patch, inserted {ver_str} into {file_path}")
        exit(0)

