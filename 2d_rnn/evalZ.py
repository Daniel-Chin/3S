from os import path
from itertools import count

from PIL import Image, ImageDraw, ImageFont
import torch

from makeDataset import (
    RESOLUTION, drawBall, rasterize, 
)
from loadDataset import img2Tensor
from vae import VAE
from main import (
    RAND_INIT_TIMES, EXPERIMENTS, renderExperimentPath, 
)

N_CURVES = 5
N_CURVE_SEGMENTS = 20
EVAL_RADIUS = 1.7
EPOCH_INTERVAL = 2

CELL_RESOLUTION = 256
HEADING_ROW_HEIGHT = 0.3

n_curve_vertices = N_CURVE_SEGMENTS + 1
FONT = ImageFont.truetype("verdana.ttf", 24)

def main():
    frames = numerateEpochs()
    frames[0].save(
        'evalZ.gif', save_all=True, append_images=frames[1:], 
        duration=200, loop=0, 
    )
    print('written to GIF.')

def numerateEpochs():
    eval_data, _ = genEvalData()
    frames = []
    vae = VAE()
    for epoch in count(0, EPOCH_INTERVAL):
        print(epoch)
        frame = Image.new('RGB', (
            CELL_RESOLUTION * (len(EXPERIMENTS)), 
            round(CELL_RESOLUTION * (
                HEADING_ROW_HEIGHT + RAND_INIT_TIMES
            )), 
        ))
        frames.append(frame)
        imDraw = ImageDraw.Draw(frame)
        for exp_i, (name, config) in enumerate(EXPERIMENTS):
            textCell(
                imDraw, name, 
                exp_i + .5, HEADING_ROW_HEIGHT * .5, 
            )
            for rand_init_i in range(RAND_INIT_TIMES):
                try:
                    vae.load_state_dict(torch.load(path.join(
                        renderExperimentPath(
                            rand_init_i, config, 
                        ), f'{epoch}_vae.pt', 
                    )))
                except FileNotFoundError:
                    print('epoch', epoch, 'not found.')
                    return frames[:-1]
                vae.eval()
                with torch.no_grad():
                    z, _ = vae.encode(eval_data)
                    drawCell(
                        imDraw, z, exp_i, 
                        rand_init_i + HEADING_ROW_HEIGHT, 
                    )

def textCell(imDraw, text, col_i, row_i):
    imDraw.text((
        col_i * CELL_RESOLUTION, 
        row_i * CELL_RESOLUTION, 
    ), text, font=FONT, anchor='mm')

def drawCell(imDraw, z, col_i, row_i):
    x_offset = CELL_RESOLUTION * col_i
    y_offset = CELL_RESOLUTION * row_i
    for curve_i in range(N_CURVES * 2):
        z_seg = z[
            n_curve_vertices * curve_i : 
            n_curve_vertices * (curve_i + 1), 
            :
        ]
        imDraw.line(
            [
                (
                    rasterize(
                        z_seg[i, 0].item(), 
                        4, CELL_RESOLUTION, 
                    ) + x_offset, 
                    rasterize(
                        z_seg[i, 1].item(), 
                        4, CELL_RESOLUTION, 
                    ) + y_offset, 
                ) 
                for i in range(n_curve_vertices)
            ], 
            ('red', 'green')[curve_i % 2], 
        )

def genEvalData():
    n_datapoints = N_CURVES * n_curve_vertices * 2
    eval_data = torch.zeros(
        n_datapoints, 1, 
        RESOLUTION, RESOLUTION, 
    )
    z_truth = torch.zeros(n_datapoints, 2)
    X = []
    Y = []
    for curve_i in range(N_CURVES):
        curve_pos = (
            curve_i / (N_CURVES - 1) - .5
        ) * 2 * EVAL_RADIUS
        for vertex_i in range(n_curve_vertices):
            vertex_pos = ((
                vertex_i / N_CURVE_SEGMENTS - .5
            ) * 2 * EVAL_RADIUS)

            data_i = (2 * curve_i    ) * n_curve_vertices + vertex_i
            eval_data[data_i, 0, :, :] = img2Tensor(
                drawBall(curve_pos, vertex_pos), 
            )
            z_truth[data_i, 0] = curve_pos
            z_truth[data_i, 1] = vertex_pos

            data_i = (2 * curve_i + 1) * n_curve_vertices + vertex_i
            eval_data[data_i, 0, :, :] = img2Tensor(
                drawBall(vertex_pos, curve_pos), 
            )
            z_truth[data_i, 0] = vertex_pos
            z_truth[data_i, 1] = curve_pos

            X.append(vertex_pos)
            Y.append(curve_pos)
    print('# of eval points:', n_datapoints)
    # plt.scatter(X, Y)
    # plt.show()
    return eval_data, z_truth

main()
