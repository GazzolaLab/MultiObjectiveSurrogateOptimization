import os
from miv_simulator import mechanisms
import shutil


def find_directories_with_mechanisms(root_dir):
    directories_with_mechanisms = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if "mechanisms" in dirnames:
            directories_with_mechanisms.append(dirpath)
    return directories_with_mechanisms


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    result = find_directories_with_mechanisms(script_dir)
    for directory in result:
        compiled_dir = os.path.join(directory, "mechanisms", "compiled")
        if os.path.exists(compiled_dir):
            shutil.rmtree(compiled_dir)
        print(directory)
        mechanisms.compile(os.path.join(directory, "mechanisms"), force=True)
