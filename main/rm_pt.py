import os
from os import path

from tqdm import tqdm

def main():
    print('This will remove all vae model checkpoints except the last epoch. ')
    assert input('continue? y/n >').lower() == 'y'
    assert input('REALLY continue? y/n >').lower() == 'y'
    os.chdir('experiments')
    for exp in (tqd := tqdm(os.listdir())):
        tqd.set_description(exp)
        if not path.isdir(exp):
            continue
        os.chdir(exp)
        for exp in os.listdir():
            if not path.isdir(exp):
                continue
            if exp == '__pycache__':
                continue
            os.chdir(exp)
            clean()
            os.chdir('..')
        os.chdir('..')

template = 'vae_epoch_%d.pt'
prefix, suffix = template.split('%d')
def clean():
    files = os.listdir()
    max_epoch = -1
    all_vae_pts = []
    for file in files:
        if not file.startswith(prefix):
            continue
        all_vae_pts.append(file)
        _, file = file.split(prefix)
        file, _ = file.split(suffix)
        epoch = int(file)
        max_epoch = max(max_epoch, epoch)
    assert max_epoch != -1
    negative = template % max_epoch
    # print(*all_vae_pts, sep='\n')
    # print()
    for vae_pt in all_vae_pts:
        if vae_pt == negative:
            continue
        # input(vae_pt)
        os.remove(vae_pt)

main()
