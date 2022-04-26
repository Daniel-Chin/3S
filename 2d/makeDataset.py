import os
import pickle
import numpy as np
from PIL import Image, ImageDraw

N_DATAPOINTS = 10
PATH = './dataset'
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

def rasterize(x):
    return round((x + CANVAS_RADIUS) / (
        CANVAS_RADIUS * 2
    ) * RESOLUTION)

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
    os.chdir(PATH)
    root = []
    for i in range(N_DATAPOINTS):
        x, y = sampleXY()
        canvas = drawBall(x, y)
        filename = f'{i}.png'
        canvas.save(filename)
        root.append((filename, (x, y)))
    with open('root.pickle', 'wb') as f:
        pickle.dump(root, f)

if __name__ == '__main__':
    main()
