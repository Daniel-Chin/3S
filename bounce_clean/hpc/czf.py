# create tar.gz for experiments

import os
from os import path

def main():
    os.chdir('../experiments')
    list_dir = os.listdir()
    all_gz = set()
    all_dir = set()
    IGNORE = ['.gitignore', '.', '..']
    for node in list_dir:
        if node in IGNORE:
            continue
        base, ext = path.splitext(node)
        print(base, ext)
        if path.isdir(node):
            all_dir.add(path.normpath(node))
        elif ext.lower() == '.gz':
            all_gz.add(path.normpath(base))
        else:
            print('Warning: unknown file:', node)
    for dir in all_dir:
        if dir not in all_gz:
            os.system(f'tar -vczf {dir}')

main()
