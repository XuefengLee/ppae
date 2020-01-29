import torch
import torch.nn as nn
#import torch.nn.functional as F
import pdb

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class cnn(nn.Module):

    def __init__(self, z_dim=64, nc=3):
        super(cnn, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 128, 4, 2, 1, bias=False),  # 14,  128, 32, 32
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),  # 7,  256, 16, 16
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),  # B,  512,  8,  8
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),  # B, 1024,  4,  4
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
        )
        self.fc = nn.Linear(128 * (2 ** 3), self.z_dim)

        #
        #     View((-1, 1024 * 4 * 4)),  # B, 1024*4*4
        #     nn.Linear(1024 * 4 * 4, z_dim)  # B, z_dim
        # )

        self.proj = nn.Sequential(
            nn.Linear(self.z_dim, 128 * 8 * 7 * 7),
            nn.ReLU()
        )

        # nn.Linear(z_dim, 1024 * 8 * 8),  # B, 1024*8*8
        # View((-1, 1024, 8, 8)),  # B, 1024,  8,  8

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(1024, 512, 4,bias=False),  # B,  512, 16, 16
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, bias=False),  # B,  256, 32, 32
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 1, 4, 2,bias=False),  # B,  128, 64, 64
            # nn.BatchNorm2d(128),
            # nn.ReLU(True),
            # nn.ConvTranspose2d(128, nc, 1),  # B,   nc, 64, 64
            nn.Sigmoid()
        )

    def encode(self, x):
        z = self.encoder(x)
        z = z.squeeze()
        z = self.fc(z)

        return z

    def decode(self, z):
        z = self.proj(z)
        z = z.view(-1, 128 * 8, 7, 7)
        recon = self.decoder(z)

        return recon

    def recon(self,x):
        z = self.encode(x)
        r = self.decode(z)

        return r

class mlp(nn.Module):

    def __init__(self, z_dim=20):
        super(mlp, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(784, 400),
            nn.ReLU(True),
            nn.Linear(400, z_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 400),
            nn.ReLU(True),
            nn.Linear(400, 784),
            nn.Sigmoid()
        )

    def encode(self, x):
        z = self.encoder(x.view(-1, 784))

        return z

    def decode(self, z):

        recon = self.decoder(z).view(-1,28,28)

        return recon

    def recon(self,x):
        z = self.encoder(x.view(-1, 784))
        r = self.decoder(z).view(-1, 28, 28)

        return r
