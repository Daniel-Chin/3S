from math import sin

class Constant:
    def __init__(self, value) -> None:
        self.value = value

    def __call__(self, _):
        return self.value
    
    def __repr__(self):
        return str(self.value)

class WarmUp:
    def __call__(self, epoch):
        if epoch < 500:
            return 0.002 * (epoch / 500)
        else:
            return 0.002
    
    def __repr__(self):
        return 'warm_up'

class Osc:
    def __call__(self, epoch):
        if epoch < 500:
            return 0.002 * (epoch / 500)
        else:
            return 0.002 + sin((epoch - 500) * .03) * 0.002
    
    def __repr__(self):
        return 'osc'

def test():
    from matplotlib import pyplot as plt
    import numpy as np
    x = np.linspace(0, 1300, 200)
    osc = Osc()
    y = np.zeros_like(x)
    for i, x_i in enumerate(x):
        y[i] = osc(x_i)
    plt.plot(x, y)
    plt.show()

if __name__ == '__main__':
    test()
