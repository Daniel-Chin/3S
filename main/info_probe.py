import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam

from shared import *
from vae import VAE
from load_dataset import Dataset, dataLoader

class InfoProbeNetwork(nn.Module):
    def __init__(
        self, experiment, hyperParams: HyperParams, 
    ) -> None:
        super().__init__()
        fcs = []
        c0 = hyperParams.symm.latent_dim
        for c1 in hyperParams.vae_channels:
            fcs.append(nn.Linear(c0, c1))
            fcs.append(nn.ReLU())
            c0 = c1
        fcs.append(nn.Linear(c0, experiment.DATASET_INSTANCE.ACTUAL_DIM))
        self._forward = nn.Sequential(*fcs)
    
    def forward(self, /, x):
        return self._forward(x)

class InfoProbeDataset(torch.utils.data.Dataset):
    def __init__(
        self, experiment, hParams: HyperParams, 
        vae: VAE, 
        dataSet: Dataset, 
    ):
        SEQ_LEN    = experiment.DATASET_INSTANCE.SEQ_LEN
        vae.eval()
        z = []
        traj = []
        with torch.no_grad(), hParams.eval():
            loader = dataLoader(
                dataSet, hParams.batch_size, dataSet.size, 
            )
            for video_batch, traj_batch in loader:
                flat_video_batch = video_batch.view(
                    hParams.batch_size * SEQ_LEN, 
                    experiment.DATASET_INSTANCE.IMG_N_CHANNELS, 
                    *hParams.signal_resolution, 
                )
                flat_traj_batch = traj_batch.view(
                    hParams.batch_size * SEQ_LEN, -1, 
                )
                mu, _ = vae.encode(flat_video_batch)
                z.append(mu)
                traj.append(flat_traj_batch)
            self.z = torch.concat(z).detach()
            self.traj = torch.concat(traj).detach()
            self.traj = self.traj - self.traj.mean(dim=0)
            self.traj = self.traj / self.traj.std (dim=0)
        self.size = self.z.shape[0]
    
    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return (
            self.z   [index, :], 
            self.traj[index, :], 
        )

def probe(
    experiment, hParams: HyperParams, 
    trainSet: InfoProbeDataset, validateSet: InfoProbeDataset, 
    n_epochs: int, 
):
    probeNetwork = InfoProbeNetwork(experiment, hParams)
    adam = Adam(probeNetwork.parameters(), 1e-3)

    train_losses = []
    validate_losses = []
    for epoch in range(n_epochs):
        trainLoader    = dataLoader(
            trainSet, 512, None, 
        )
        with hParams.eval():
            validateLoader = dataLoader(
                validateSet, 512, None, 
            )
        
        probeNetwork.train()
        in_epoch_train_losses = []
        for z, traj in trainLoader:
            traj_hat = probeNetwork.forward(z)
            train_loss = F.mse_loss(traj_hat, traj)
            adam.zero_grad()
            train_loss.backward()
            adam.step()
            in_epoch_train_losses.append(train_loss.detach())
        epoch_train_loss = torch.stack(
            in_epoch_train_losses, 
        ).detach().mean().cpu()

        probeNetwork.eval()
        with torch.no_grad():
            for z, traj in validateLoader:
                traj_hat = probeNetwork.forward(z)
                validate_loss = F.mse_loss(traj_hat, traj).cpu()
        
        train_losses.append(epoch_train_loss)
        validate_losses.append(validate_loss)
    return train_losses, validate_losses
