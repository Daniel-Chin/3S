from os import path

from PIL import Image, ImageDraw
import torch

from makeDataset import (
    RESOLUTION, drawBall, CANVAS_RADIUS, rasterize, 
)
from loadDataset import img2Tensor
from vae import VAE
from train import CHECKPOINTS_PATH, N_EPOCHS
from main import RAND_INIT, BETAS

CELL_RESOLUTION = 256
N_CURVES = 5
N_CURVE_SEGMENTS = 10

n_curve_vertices = N_CURVE_SEGMENTS + 1

def main():
    eval_data = genEvalData()
    frames = []
    for epoch in range(N_EPOCHS):
        frame = Image.new('L', (
            CELL_RESOLUTION * (1 + BETAS), 
            CELL_RESOLUTION * (1 + RAND_INIT), 
        ))
        frames.append(frame)
        imDraw = ImageDraw.Draw(frame)
        for rand_init_i in range(RAND_INIT):
            for beta_i, beta in enumerate(BETAS):
                vae = VAE()
                vae.load_state_dict(torch.load(path.join(
                    CHECKPOINTS_PATH, f'{beta}_{rand_init_i}', 
                    f'{epoch}.pt', 
                )))
                vae.eval()
                with torch.no_grad():
                    z, _ = vae.encode(eval_data)
                    drawCell(imDraw, z, beta_i, rand_init_i)

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
                        z_seg[i, 0], CELL_RESOLUTION, 
                    ) + x_offset, 
                    rasterize(
                        z_seg[i, 1], CELL_RESOLUTION, 
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
    for curve_i in range(N_CURVES):
        curve_pos = curve_i / (N_CURVES + 1) * CANVAS_RADIUS
        for vertex_i in range(n_curve_vertices):
            vertex_pos = (
                vertex_i / N_CURVE_SEGMENTS * CANVAS_RADIUS
            )
            eval_data[
                (2 * curve_i    ) * n_curve_vertices + vertex_i, 
                0, :, :, 
            ] = img2Tensor(drawBall(curve_pos, vertex_pos))
            eval_data[
                (2 * curve_i + 1) * n_curve_vertices + vertex_i, 
                0, :, :, 
            ] = img2Tensor(drawBall(vertex_pos, curve_pos))
    print('# of eval points:', n_datapoints)
    return eval_data

main()
