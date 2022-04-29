import os
from os import path
import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from PIL import Image, ImageDraw

TRAIN_SET_SIZE = 256
VALIDATE_SET_SIZE = 64
PATH = './dataset'
TRAIN_PATH    = path.join(PATH, 'train')
VALIDATE_PATH = path.join(PATH, 'validate')
CANVAS_RADIUS = 2
BALL_RADIUS = .3
RESOLUTION = 32

def sampleXY():
    while True:
        x = np.random.standard_normal()
        y = np.random.standard_normal()
        if (
            abs(x) + BALL_RADIUS < CANVAS_RADIUS and 
            abs(y) + BALL_RADIUS < CANVAS_RADIUS
        ):
            break
    return x, y

def rasterize(x, resolution=RESOLUTION):
    return round((x + CANVAS_RADIUS) / (
        CANVAS_RADIUS * 2
    ) * resolution)

def drawBall(x, y):
    canvas = Image.new('RGB', (RESOLUTION, RESOLUTION))
    draw = ImageDraw.Draw(canvas)
    draw.ellipse((
        rasterize(x - BALL_RADIUS), 
        rasterize(y - BALL_RADIUS), 
        rasterize(x + BALL_RADIUS), 
        rasterize(y + BALL_RADIUS), 
    ), fill = 'green', outline ='green')
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
        x, y = sampleXY()
        canvas = drawBall(x, y)
        filename = f'{i}.png'
        canvas.save(filename)
        root.append((filename, (x, y)))
        print(i)
    with open('root.pickle', 'wb') as f:
        pickle.dump(root, f)
    ax = plt.gca()
    for _, (x, y) in root:
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
