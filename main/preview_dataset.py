from matplotlib import pyplot as plt

from shared import torch2PIL
from load_dataset import VideoDataset, MusicDataset
from dataset_instances import IonianScales_fr3gm

def video():
    dataset = VideoDataset(
        '../datasets/xj_bounce/validate', 64, 20, 3, 32, 3, 
    )
    for video, traj in dataset:
        plt.plot(traj)
        plt.show()

def music():
    dataset = MusicDataset(
        IonianScales_fr3gm.songBox, IonianScales_fr3gm.config, 
        True, 8, 
    )
    for video, traj in dataset:
        T, _, freqs, wid = video.shape
        # fig, axes = plt.subplots(1, T)
        # for t, ax in enumerate(axes):
        #     spec = video[t, 0, :, :]
        #     ax.imshow(spec)
        a = video[:, 0, :, :].permute(0, 2, 1)
        b = a.reshape(T * wid, freqs)
        plt.imshow(b.T)
        plt.gca().invert_yaxis()
        plt.show()

# video()
music()
