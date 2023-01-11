# commit exps, and create tar.gz for experiments

import os
from os import path
import subprocess as sp

def main():
    os.chdir('../experiments')
    os.system('git status')
    print('Make sure working tree is clean!!!')
    print('(Enter empty string to begin.)')
    exps = []
    do_continue = True
    while do_continue:
        do_continue = False
        for op in input('exp_dir_name = ').split('\n'):
            op = op.lstrip('#').strip(' /\\\n')
            if op == '':
                continue
            do_continue = True
            if path.isdir(op):
                exps.append(op)
            else:
                print('Not a dir. ')
    print('start...')
    if exps:
        for exp in exps:
            doOne(exp)
    else:
        if input('Do all? y/n: ').lower() == 'y':
            doAll()
        else:
            print('did nothing.')
            return
    os.system('git commit -m "auto commit exp"')
    os.system('git push')

template = 'vae_epoch_%d.pt'
prefix, suffix = template.split('%d')
def tar(exp_dir_name):
    print('taring', exp_dir_name, '...')
    with sp.Popen(
        ['tar', '-czf', exp_dir_name + '.tar.gz', '-T', '-'], 
        stdin=sp.PIPE, 
    ) as p:
        paths = [exp_dir_name]
        def eat(name: str):
            p.stdin.write(path.join(*paths, name).encode())
            p.stdin.write('\n'.encode())

        for name in os.listdir(exp_dir_name):
            if path.isfile(path.join(*paths, name)):
                eat(name)
            else:
                if name == '__pycache__':
                    continue
                # print('  doing', name)
                paths.append(name)
                max_epoch = -1
                for name in os.listdir(path.join(*paths)):
                    if not name.startswith(prefix):
                        eat(name)
                        continue
                    _, name = name.split(prefix)
                    name, _ = name.split(suffix)
                    epoch = int(name)
                    max_epoch = max(max_epoch, epoch)
                assert max_epoch != -1
                eat(template % max_epoch)
                paths.pop(-1)
        p.stdin.close()
        # print('  waiting...')
        p.wait()
    # print('  exit')

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
