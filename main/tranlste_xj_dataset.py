import os
from os import path
import ast
import shutil
import json

from tqdm import tqdm

SRC = '../../xjRepo/codes/S3Ball/Ball3DImg/32_32_0.2_20_3_init_points_subset_2048/'
DEsT = '../datasets/xj_bounce'

def main():
    all_trajs = os.listdir(SRC)
    assert len(all_trajs) == 2048
    TRAIN_SIZE = 1024
    VALIDATE_SIZE = 64
    train_trajs = all_trajs[:TRAIN_SIZE]
    all_trajs = all_trajs[TRAIN_SIZE:]
    valdidate_trajs = all_trajs[:VALIDATE_SIZE]
    for set_name, traj_names in (
        ('train', train_trajs), 
        ('validate', valdidate_trajs), 
    ):
        for traj_i, dir_name in enumerate(tqdm(traj_names, set_name)):
            src_cd = path.join(SRC, dir_name)
            filenames = os.listdir(src_cd)
            index = []
            for fn in filenames:
                base, _ = path.splitext(fn)
                t, pos = base.split('.', 1)
                t = int(t)
                pos = ast.literal_eval(pos)
                index.append((t, fn, pos))
            index.sort(key = lambda x : x[0])
            dest_cd = path.join(DEsT, set_name, str(traj_i))
            os.makedirs(dest_cd, exist_ok=True)
            traj = []
            for t, fn, pos in index:
                velocity = [0, 0, 0]
                traj.append([[pos, velocity]])
                shutil.copy(
                    path.join(src_cd, fn), 
                    path.join(dest_cd, f'{t}.png')
                )
            with open(path.join(dest_cd, 'trajectory.json'), 'w') as f:
                json.dump(traj, f)

main()
