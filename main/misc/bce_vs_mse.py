import numpy as np
from matplotlib import pyplot as plt

# BCE: - y ⋅ log(x) - (1−y) ⋅ log(1−x) 
# MSE: (x - y)^2

def gradBCE(x, y):
    return - y * (1/x) - (1-y) * (-1/(1-x))

def gradMSE(x, y):
    return 2 * (x - y)

def main():
    x = np.linspace(0, 1, 100)
    for y, c in (
        (0.0, 'r'), 
        (0.5, 'g'), 
        (1.0, 'b'), 
    ):
        for loss, display, style in (
            (gradBCE, 'BCE', ':'), 
            (lambda x, y : (gradMSE(x, y) * 2), 'MSE*2', '-'), 
        ):
            plt.plot(x, loss(x, y), c + style, label=f'{display}, {y=}')
    plt.legend()
    plt.ylim(-3, 3)
    plt.xlabel('x')
    plt.ylabel('Loss Gradient')
    plt.show()

main()

'''
Conclusion
2 * MSE can replace BCE, I think. 
'''
