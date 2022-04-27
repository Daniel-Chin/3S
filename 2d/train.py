import os
from os import path
import numpy as np
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
from makeDataset import drawBall, RESOLUTION

N_EPOCHS = 1000
BETA = torch.tensor(.3)
CHANNELS = [8, 16, 32]

RECONSTRUCT_PATH = './reconstruct'
COORD_RADIUS = 2
TERMINAL_RESOLUTION = 20

has_cuda = torch.cuda.is_available()
if has_cuda:
    device = torch.device("cuda:0")
    print('We have CUDA.')
else:
    device = torch.device("cpu")
    print("We DON'T have CUDA.")

class Encoder(nn.Module):
    def __init__(self) -> None:
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
        self.fc = nn.Linear(self.fc_in, 2)
    
    def forward(self, x):
        x = F.relu(self.conv_0(x))
        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
        x = x.view((x.shape[0], self.fc_in))
        x = self.fc(x)
        return x

class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # self.fc_0 = nn.Linear(2, 16)
        # self.fc_1 = nn.Linear(16, 32)
        # self.fc_2 = nn.Linear(32, 64)
        # self.fc_3 = nn.Linear(64, CHANNELS[-1] * 4 * 4)

        # self.fc_0 = nn.Linear(2, 32)
        # self.fc_1 = nn.Linear(32, CHANNELS[-1] * 4 * 4)

        self.fc = nn.Linear(2, CHANNELS[-1] * 4 * 4)

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
    
    def forward(self, x):
        # x = F.relu(self.fc_0(x))
        # x = F.relu(self.fc_1(x))
        # x = F.relu(self.fc_2(x))
        # x = self.fc_3(x)

        # x = F.relu(self.fc_0(x))
        # x = self.fc_1(x)

        x = self.fc(x)

        x = x.view((x.shape[0], CHANNELS[-1], 4,  4))
        x = F.relu(self.deconv_0(x))
        x = F.relu(self.deconv_1(x))
        x = torch.sigmoid(self.deconv_2(x))
        return x

def loadModels():
    return Encoder(), Decoder()

def main():
    encoder, decoder = loadModels()
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
        # train encoder
        z = encoder(images)
        encoder_loss = mse(z, coordinates)
        encoderOptim.zero_grad()
        encoder_loss.backward()
        encoderOptim.step()

        # train decoder
        noise = torch.normal(
            mean=torch.zeros((batch_size, 2)), 
            std=BETA, 
        ).to(device)
        reconstruct = decoder(coordinates + noise)
        decoder_loss = mse(reconstruct, images)
        decoderOptim.zero_grad()
        decoder_loss.backward()
        decoderOptim.step()

        if epoch % 10 == 0:
            print('Finished epoch', epoch)
            print('  Encoder loss', encoder_loss.item())
            print('  Decoder loss', decoder_loss.item())
    with torch.no_grad():
        evaluate(
            encoder.cpu(), decoder.cpu(), 
            images.cpu(), coordinates.cpu(), 
        )

def render(shot, aimed, datapoints):
    def rasterize(coord):
        x, y = (
            coord / COORD_RADIUS + 1
        ) / 2 * TERMINAL_RESOLUTION
        return int(x.item()), int(y.item())
    buffer = [
        [' '] * TERMINAL_RESOLUTION 
        for _ in range(TERMINAL_RESOLUTION)
    ]
    for i in range(datapoints.shape[0]):
        x, y = rasterize(datapoints[i, :])
        buffer[y][x] = '-'
    x, y = rasterize(shot)
    buffer[y][x] = 'X'
    x, y = rasterize(aimed)
    buffer[y][x] = '!' if buffer[y][x] == 'X' else 'O'
    for line in buffer:
        print(*line, sep='')

def evaluate(encoder, decoder, images, coordinates):
    # eval encoder
    z = encoder(images)
    for i in range(images.shape[0]):
        coord = coordinates[i, :]
        render(z[i, :], coord, torch.Tensor())
        print()
        input('Enter...')
    target_grid = np.ones((2, )) * (TERMINAL_RESOLUTION // 2)
    while True:
        target = (
            target_grid / TERMINAL_RESOLUTION - .5
        ) * COORD_RADIUS * 2
        np_img = np.array(drawBall(*target))[:, :, 1]
        torch_img = (
            torch.from_numpy(np_img).float() / 128
        ).unsqueeze(0)
        z = encoder(torch_img.unsqueeze(0))
        render(z[0, :], target, coordinates)
        print('Use "WASD" to move the target, "ESC" to quit.')
        op = listen(b'wasd\x1b', priorize_esc_or_arrow=True)
        if op == b'w':
            target_grid[1] -= 1
            if target_grid[1] < 0:
                target_grid[1] = 0
        elif op == b's':
            target_grid[1] += 1
            if target_grid[1] >= TERMINAL_RESOLUTION:
                target_grid[1] = TERMINAL_RESOLUTION - 1
        elif op == b'a':
            target_grid[0] -= 1
            if target_grid[0] < 0:
                target_grid[0] = 0
        elif op == b'd':
            target_grid[0] += 1
            if target_grid[0] >= TERMINAL_RESOLUTION:
                target_grid[0] = TERMINAL_RESOLUTION - 1
        elif op == b'\x1b':
            break

    # evel decoder
    reconstructs = decoder(coordinates)
    os.makedirs(RECONSTRUCT_PATH, exist_ok=True)
    def visualize(reconstruct):
        return Image.fromarray(
            reconstruct.numpy() * 128
        ).convert('L')
    imgs = []
    for i in range(images.shape[0]):
        img = visualize(reconstructs[i, 0, :, :])
        img.save(path.join(
            RECONSTRUCT_PATH, f'{i}.png', 
        ))
        imgs.append(img)
    print(f'Saved reconstruction to `{RECONSTRUCT_PATH}`')

    for i in range(images.shape[0] - 1):
        j = i + 1
        coord = (coordinates[i, :] + coordinates[j, :]) / 2
        reconstructs = decoder(coord.unsqueeze(0))
        img = visualize(reconstructs[0, 0, :, :])
        imgs[i].save(
            path.join(
                RECONSTRUCT_PATH, f'interpolate_{i}_{j}.gif', 
            ), save_all=True, append_images=[img, imgs[j]], 
            duration=500, loop=0, 
        )
    print(f'Saved interpolation to `{RECONSTRUCT_PATH}`')

main()
