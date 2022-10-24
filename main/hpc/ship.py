# commit exps, and create tar.gz for experiments

import os
from os import path

def main():
    os.chdir('../experiments')
    os.system('git status')
    print('Make sure working tree is clean!!!')
    print('(Enter empty string to begin.)')
    exps = []
    while True:
        op = input('exp_dir_name = ')
        if op == '':
            break
        if path.isdir(op):
            exps.append(op)
        else:
            print('Not a dir. ')
    if exps:
        for exp in exps:
            doOne(exp)
    else:
        if input('Do all? y/n: ').lower == 'y':
            doAll()
        else:
            print('did nothing.')
            return
    os.system('git commit -m "auto commit exp"')
    os.system('git push')

def tar(name):
    os.system(f'tar -czf "{name}.tar.gz" "{name}"')

def doAll():
    os.system('git add .')
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
            _base, tar = path.splitext(base)
            all_gz.add(path.normpath(_base))
        else:
            print('Warning: unknown file:', node)
    print(all_dir)
    print(all_gz)
    for dir in all_dir:
        if dir not in all_gz:
            tar(dir)

def doOne(exp_dir_name: str):
    os.system('git add ' + exp_dir_name)
    tar(exp_dir_name)

main()
