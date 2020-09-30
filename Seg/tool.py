import os
import fnmatch


def find_recursive(root_dir, ext='.jpg'):
    files = []
    for root, dirnames, filenames in os.walk(root_dir):
        for filename in fnmatch.filter(filenames, '*' + ext):
            files.append(filename)
    return files
root_dir = "../proj/data/emotic19/emotic/mscoco/images"
print(" ".join(find_recursive(root_dir)))