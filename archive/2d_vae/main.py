from train import train

RAND_INIT = 4
BETAS = (0, 0.0001, 0.0003, 0.001, 0.002)

def main():
    for rand_init_i in range(RAND_INIT):
        for beta in BETAS:
            train(beta, rand_init_i)

if __name__ == '__main__':
    main()
