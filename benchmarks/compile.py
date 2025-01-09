import os
from miv_simulator import mechanisms


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
        print(directory)
        mechanisms.compile(directory, force=True)
