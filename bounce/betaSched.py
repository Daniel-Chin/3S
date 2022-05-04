import numpy as np


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
    def __init__(
        self, middle=0.002, slow_start=500, osc_amp=0.002, 
        period=200, 
    ):
        self.middle = middle
        self.slow_start = slow_start
        self.osc_amp = osc_amp
        self.period = period
    
    def __call__(self, epoch):
        if epoch < self.slow_start:
            return self.middle * (epoch / self.slow_start)
        else:
            return self.middle + np.sin(
                (epoch - self.slow_start) / 200 * 2 * np.pi
            ) * self.osc_amp
    
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
