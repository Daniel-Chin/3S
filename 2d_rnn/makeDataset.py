import os
from os import path
import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from PIL import Image, ImageDraw

TRAIN_SET_SIZE = 256
VALIDATE_SET_SIZE = 64
SEQ_LEN = 10
CANVAS_RADIUS = 2
BALL_RADIUS = .3
V_STD = .1
RESOLUTION = 32
PATH = './dataset'
TRAIN_PATH    = path.join(PATH, 'train')
VALIDATE_PATH = path.join(PATH, 'validate')

def inBound(x):
    return abs(x) + BALL_RADIUS < CANVAS_RADIUS

def sample():
    while True:
        x = np.random.standard_normal()
        y = np.random.standard_normal()
        v_x = np.random.standard_normal() * V_STD
        v_y = np.random.standard_normal() * V_STD
        if (v_x**2 + v_y**2)**.5 < .05:
            print('slow')
            continue
        if not inBound(x):
            print('start')
            continue
        if not inBound(y):
            print('start')
            continue
        if not inBound(x + SEQ_LEN * v_x):
            print('end')
            continue
        if not inBound(y + SEQ_LEN * v_y):
            print('end')
            continue
        return x, y, v_x, v_y

def rasterize(x, x_radius=CANVAS_RADIUS, resolution=RESOLUTION):
    return round((x + x_radius) / (
        x_radius * 2
    ) * resolution)

def drawBall(x, y):
    canvas = Image.new('L', (RESOLUTION, RESOLUTION))
    draw = ImageDraw.Draw(canvas)
    draw.ellipse((
        rasterize(x - BALL_RADIUS), 
        rasterize(y - BALL_RADIUS), 
        rasterize(x + BALL_RADIUS), 
        rasterize(y + BALL_RADIUS), 
    ), fill = 'white', outline ='white')
    return canvas

def main():
    os.makedirs(   TRAIN_PATH, exist_ok=True)
    os.chdir(TRAIN_PATH)
    makeOneSet(TRAIN_SET_SIZE)
    os.chdir('../..')
    os.makedirs(VALIDATE_PATH, exist_ok=True)
    os.chdir(VALIDATE_PATH)
    makeOneSet(VALIDATE_SET_SIZE)

def makeOneSet(set_size):
    root = []
    for i in range(set_size):
        x, y, v_x, v_y = sample()
        frames = []
        trajectory = []
        for _ in range(SEQ_LEN):
            frame = drawBall(x, y)
            frames.append(frame)
            trajectory.append((x, y))
            x += v_x
            y += v_y
        filename = f'{i}.gif'
        frames[0].save(
            filename, save_all=True, append_images=frames[1:], 
            duration=200, loop=0, 
        )
        root.append((filename, trajectory))
        print(i)
    with open('root.pickle', 'wb') as f:
        pickle.dump(root, f)
    ax = plt.gca()
    for _, trajectory in root:
        for x, y in trajectory:
            circle = Circle((x, y), BALL_RADIUS, color='g')
            ax.add_patch(circle)
    ax.axis('equal')
    ax.axis((
        -CANVAS_RADIUS, CANVAS_RADIUS, 
        -CANVAS_RADIUS, CANVAS_RADIUS, 
    ))
    plt.show()

if __name__ == '__main__':
    main()
