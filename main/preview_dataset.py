from matplotlib import pyplot as plt

from load_dataset import VideoDataset

def main():
    dataset = VideoDataset(
        '../datasets/xj_bounce/validate', 64, 20, 3, 32, 
    )
    for video, traj in dataset:
        plt.plot(traj)
        plt.show()

main()
