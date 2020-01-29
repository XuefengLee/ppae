import torch
import torch.nn as nn
#import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

class Encoder(nn.Module):
    """Encoder-Decoder architecture for both WAE-MMD and WAE-GAN."""

    def __init__(self, z_dim=64, nc=3):
        super(Encoder, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 128, 4, 2, 1, bias=False),  # B,  128, 32, 32
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),  # B,  256, 16, 16
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),  # B,  512,  8,  8
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),  # B, 1024,  4,  4
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            View((-1, 1024 * 4 * 4)),  # B, 1024*4*4
            nn.Linear(1024 * 4 * 4, z_dim)  # B, z_dim
        )

    def forward(self, x):
        z = self.encoder(x)

        return z

class Decoder(nn.Module):
    """Encoder-Decoder architecture for both WAE-MMD and WAE-GAN."""
    def __init__(self, z_dim=64, nc=3):
        super(Decoder, self).__init__()
        self.z_dim = z_dim
        self.nc = nc


        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 1024 * 8 * 8),  # B, 1024*8*8
            View((-1, 1024, 8, 8)),  # B, 1024,  8,  8
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),  # B,  512, 16, 16
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),  # B,  256, 32, 32
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),  # B,  128, 64, 64
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, nc, 1),  # B,   nc, 64, 64
            nn.Sigmoid()
        )

    def forward(self, z):

        x_recon = self.decoder(z)

        return x_recon



if __name__ == '__main__':
    pass