from matplotlib import pyplot as plt

from load_dataset import Dataset

def main():
    dataset = Dataset(
        '../datasets/xj_bounce/validate', 64, 20, 3, 
    )
    for video, traj in dataset:
        plt.plot(traj)
        plt.show()

main()
