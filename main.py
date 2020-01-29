import argparse
import torch
import os
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.datasets import CelebA
from torchvision.transforms import transforms
from torchvision.utils import save_image
from torch.optim.lr_scheduler import StepLR
from model import Encoder,Decoder
from utils import plumGauss
import pdb

torch.manual_seed(123)

parser = argparse.ArgumentParser(description='PyTorch MNIST WAE-GAN')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('-batch_size', type=int, default=100, metavar='N', help='input batch size for training (default: 100)')
parser.add_argument('-epochs', type=int, default=110, help='number of epochs to train (default: 100)')
parser.add_argument('-lr', type=float, default=0.0001, help='learning rate (default: 0.0001)')
parser.add_argument('-dim_h', type=int, default=128, help='hidden dimension (default: 128)')
parser.add_argument('-n_z', type=int, default=64, help='hidden dimension of z (default: 64)')
parser.add_argument('-LAMBDA', type=float, default=10, help='regularization coef MMD term (default: 10)')
parser.add_argument('-n_channel', type=int, default=1, help='input channels (default: 1)')
parser.add_argument('-sigma', type=float, default=1, help='variance of hidden dimension (default: 1)')
args = parser.parse_args()


celebTrans = transforms.Compose([
    transforms.CenterCrop(140),
    transforms.Resize(64),
    transforms.ToTensor()
])
trainset = CelebA(root=args.dataroot,
                 split='train',
                 transform=celebTrans,
                 download=True)

testset = CelebA(root=args.dataroot,
                 split='test',
                 transform=celebTrans,
                 download=True)

train_loader = DataLoader(dataset=trainset,
                          batch_size=args.batch_size,
                          shuffle=True)

test_loader = DataLoader(dataset=testset,
                         batch_size=104,
                         shuffle=False)


encoder = Encoder(z_dim=64)
decoder = Decoder(z_dim=64)

criterion = nn.MSELoss()



# Optimizers
en_optim = optim.Adam(encoder.parameters(), lr = args.lr)
de_optim = optim.Adam(decoder.parameters(), lr = args.lr)


en_scheduler = StepLR(en_optim, step_size=30, gamma=0.5)
de_scheduler = StepLR(de_optim, step_size=30, gamma=0.5)


if torch.cuda.is_available():
    encoder = encoder.cuda()
    decoder = decoder.cuda()



for epoch in range(args.epochs):
    step = 0

    for images, _ in tqdm(train_loader):

        if torch.cuda.is_available():
            images = images.cuda()

        encoder.zero_grad()
        decoder.zero_grad()



        # ======== Train Generator ======== #


        batch_size = images.size()[0]


        z_real = encoder(images)
        x_recon = decoder(z_real)
        recon_loss = criterion(x_recon, images)
        loss_PP_real = plumGauss(z_real)
        loss = recon_loss + loss_PP_real


        loss.backward()
        en_optim.step()
        de_optim.step()

        step += 1

        if (step + 1) % 300 == 0:
            print(np.cov(z_real.detach().cpu().numpy()))

            print("Epoch: [%d/%d], Step: [%d/%d], Reconstruction Loss: %.4f" %
                  (epoch + 1, args.epochs, step + 1, len(train_loader), recon_loss.data.item()))

    if (epoch + 1) % 1 == 0:
        batch_size = 104
        test_iter = iter(test_loader)
        test_data = next(test_iter)

        noise = torch.randn(batch_size, args.n_z).cuda()
        samples = decoder(noise)
        #.view(batch_size, 3, 64, 64)
        reconst  = decoder(encoder(Variable(test_data[0]).cuda()))
        reconst = reconst.cpu().view(batch_size, 3, 64, 64)

        if not os.path.isdir('./data/reconst_images'):
            os.makedirs('data/reconst_images')
        if not os.path.isdir('./data/fake_images'):
            os.makedirs('data/fake_images')

        if not os.path.isdir('./data/saved_models'):
            os.makedirs('./data/saved_models')


        torch.save(encoder.state_dict(), './%s/encoder_epoch_%d.pth' % ('./data/saved_models', epoch))
        torch.save(decoder.state_dict(), './%s/decoder_epoch_%d.pth' % ('./data/saved_models', epoch))




        save_image(test_data[0].view(batch_size, 3, 64, 64), './data/reconst_images/wae_gan_input.png')
        save_image(reconst.data, './data/reconst_images/wae_gan_images_%d.png' % (epoch + 1))

        save_image(samples.data, './data/fake_images/wae_gan_images_%d.png' % (epoch + 1))