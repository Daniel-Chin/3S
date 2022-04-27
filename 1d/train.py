import os
from os import path
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image

try:
    from interactive import listen
except ImportError as e:
    module_name = str(e).split('No module named ', 1)[1].strip().strip('"\'')
    if module_name in (
        'interactive', 
    ):
        print(f'Missing module {module_name}. Please download at')
        print(f'https://github.com/Daniel-Chin/Python_Lib')
        input('Press Enter to quit...')
    raise e
from loadDataset import loadDataset
from makeDataset import (
    drawBall, RESOLUTION, CANVAS_RADIUS, sampleX, 
)

N_EPOCHS = 1000

INTERP_BETWEEN = (0, 3)
N_GIF_FRAMES = 8

has_cuda = torch.cuda.is_available()
if has_cuda:
    device = torch.device("cuda:0")
    print('We have CUDA.')
else:
    device = torch.device("cpu")
    print("We DON'T have CUDA.")

class Encoder(nn.Module):
    def __init__(self, CHANNELS) -> None:
        super().__init__()

        self.conv_0 = nn.Conv2d(
            1, CHANNELS[0], 
            kernel_size=4, stride=2, padding=1, 
        )
        self.conv_1 = nn.Conv2d(
            CHANNELS[0], CHANNELS[1], 
            kernel_size=4, stride=2, padding=1, 
        )
        self.conv_2 = nn.Conv2d(
            CHANNELS[1], CHANNELS[2], 
            kernel_size=4, stride=2, padding=1, 
        )

        self.fc_in = CHANNELS[-1] * 4 * 4
        self.fc = nn.Linear(self.fc_in, 1)
    
    def forward(self, x):
        x = F.relu(self.conv_0(x))
        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
        x = x.view((x.shape[0], self.fc_in))
        x = self.fc(x)
        return x

class Decoder(nn.Module):
    def __init__(self, CHANNELS) -> None:
        super().__init__()

        # self.fc_0 = nn.Linear(2, 16)
        # self.fc_1 = nn.Linear(16, 32)
        # self.fc_2 = nn.Linear(32, 64)
        # self.fc_3 = nn.Linear(64, CHANNELS[-1] * 4 * 4)

        # self.fc_0 = nn.Linear(2, 32)
        # self.fc_1 = nn.Linear(32, CHANNELS[-1] * 4 * 4)

        self.fc = nn.Linear(1, CHANNELS[-1] * 4 * 4)

        self.deconv_0 = nn.ConvTranspose2d(
            CHANNELS[2], CHANNELS[1], 
            kernel_size=4, stride=2, padding=1, 
        )
        self.deconv_1 = nn.ConvTranspose2d(
            CHANNELS[1], CHANNELS[0], 
            kernel_size=4, stride=2, padding=1, 
        )
        self.deconv_2 = nn.ConvTranspose2d(
            CHANNELS[0], 1, 
            kernel_size=4, stride=2, padding=1, 
        )

        self.CHANNELS = CHANNELS
    
    def forward(self, x):
        # x = F.relu(self.fc_0(x))
        # x = F.relu(self.fc_1(x))
        # x = F.relu(self.fc_2(x))
        # x = self.fc_3(x)

        # x = F.relu(self.fc_0(x))
        # x = self.fc_1(x)

        x = self.fc(x)

        x = x.view((x.shape[0], self.CHANNELS[-1], 4,  4))
        x = F.relu(self.deconv_0(x))
        x = F.relu(self.deconv_1(x))
        x = torch.sigmoid(self.deconv_2(x))
        return x

def loadModels(CHANNELS):
    return Encoder(CHANNELS), Decoder(CHANNELS)

def train(BETA=.3, CHANNELS=[8, 16, 32]):
    BETA = torch.tensor(BETA)
    encoder, decoder = loadModels(CHANNELS)
    if has_cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()
    encoder.train()
    decoder.train()
    encoderOptim = torch.optim.Adam(
        encoder.parameters(), lr=.001, 
    )
    decoderOptim = torch.optim.Adam(
        decoder.parameters(), lr=.001, 
    )
    mse = nn.MSELoss()
    images, coordinates = loadDataset(device)
    batch_size = coordinates.shape[0]
    for epoch in range(N_EPOCHS):
        encoderOptim.zero_grad()
        decoderOptim.zero_grad()

        # train encoder
        z: torch.Tensor = encoder(images)
        encoder_loss = mse(z, coordinates)
        encoder_loss.backward()

        # train decoder
        noise = torch.normal(
            mean=torch.zeros((batch_size, 1)), 
            std=BETA, 
        ).to(device)
        reconstructs = decoder(coordinates + noise)
        decoder_loss = mse(reconstructs, images)
        decoder_loss.backward()

        # cycle consistency
        z = torch.Tensor([
            sampleX() for _ in range(8)
        ]).unsqueeze(1)
        reconstructs = decoder(z)
        z_ = encoder(reconstructs)
        cycle_loss = mse(z_, z)
        cycle_loss.backward()

        encoderOptim.step()
        decoderOptim.step()

        if epoch % 100 == 0:
            print('Finished epoch', epoch)
            print('  Encoder loss', encoder_loss.item())
            print('  Decoder loss', decoder_loss.item())
            print('  Cycle loss',     cycle_loss.item())
    
    return encoder, decoder

def evaluateEncoder(encoder):
    with torch.no_grad():
        encoder_eval = [[], []]
        for target_grid in range(RESOLUTION):
            target = (
                target_grid / RESOLUTION - .5
            ) * CANVAS_RADIUS * 2
            np_img = np.array(drawBall(target))[:, :, 1]
            torch_img = (
                torch.from_numpy(np_img).float() / 128
            ).unsqueeze(0)
            z = encoder(torch_img.unsqueeze(0))
            encoder_eval[0].append(target)
            encoder_eval[1].append(z[0, :])
    return encoder_eval

def evaluateDecoder(decoder, coordinates):
    with torch.no_grad():
        eval_coords = torch.linspace(
            coordinates[INTERP_BETWEEN[0]].item(), 
            coordinates[INTERP_BETWEEN[1]].item(), 
            N_GIF_FRAMES, 
        ).unsqueeze(1)
        reconstructs = decoder(eval_coords)
        def visualize(reconstruct):
            return Image.fromarray(
                reconstruct.numpy() * 128
            ).convert('L')
        imgs = []
        for i in range(reconstructs.shape[0]):
            img = visualize(reconstructs[i, 0, :, :])
            imgs.append(img)
    return imgs

def main():
    images, coordinates = loadDataset(torch.device("cpu"))
    for channels in (
        [2, 2, 4], 
        [2, 4, 8], 
        [4, 8, 16], 
        [8, 16, 32], 
        [16, 32, 64], 
    ):
        print(f'{channels=}')
        encoder, decoder = train(CHANNELS=channels)
        plot_x, plot_y = evaluateEncoder(encoder.cpu())
        plt.plot(plot_x, plot_y, label=str(channels[-1]))
    plt.scatter(coordinates, coordinates, label='datapoints')
    plt.legend()
    plt.show()
    
    betas = (0, .1, .2, .3, .4)
    frames = [
        Image.new('L', (len(betas) * RESOLUTION, RESOLUTION)) 
        for _ in range(N_GIF_FRAMES)
    ]
    for i, beta in enumerate(betas):
        print(f'{beta=}')
        encoder, decoder = train(BETA=beta)
        imgs = evaluateDecoder(
            decoder.cpu(), coordinates.cpu(), 
        )
        for img, frame in zip(imgs, frames):
            frame.paste(
                img, (i * RESOLUTION, 0), 
            )
    filename = 'decoder.gif'
    frames[0].save(
        filename, save_all=True, append_images=frames[1:], 
        duration=300, loop=0, 
    )
    print(f'Saved interpolation to `{filename}`')

main()
