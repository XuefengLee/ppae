import argparse
import torch
import os
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
from model import celeba_model, mnist_model
from utils import plumGauss, test
from utils_data import *
import pdb

torch.manual_seed(123)

parser = argparse.ArgumentParser(description='Plum-pudding autoencoder')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--save_dir', required=True, help='path to save dir')
parser.add_argument('--batch_size', type=int, default=100, metavar='N', help='input batch size for training (default: 100)')
parser.add_argument('-epochs', type=int, default=100, help='number of epochs to train (default: 100)')
parser.add_argument('-lr', type=float, default=0.0001, help='learning rate (default: 0.0001)')
parser.add_argument('--dataset', choices=['cifar10', 'mnist', 'celeba'], type=str, help='choose dataset')
parser.add_argument('-n_z', type=int, default=64, help='hidden dimension of z (default: 64)')
parser.add_argument('--ld', type=float, default=1, help='coefficient of plum pudding loss')
parser.add_argument('--device', type=str,  default='0', help='cuda device')

args = parser.parse_args()

if not os.path.isdir(args.save_dir):
    os.makedirs(args.save_dir)
f = open(args.save_dir + "/args.txt", "w")
f.write(str(args))
f.close()

device = torch.device("cuda:" + args.device)

train_loader, test_loader = prepare_data(args.batch_size, args.dataset, args.dataroot)

autoencoder = None
if args.dataset == 'mnist':
    autoencoder = mnist_model(z_dim=args.n_z,nc=1)
elif args.dataset == 'celeba':
    autoencoder = celeba_model(z_dim=args.n_z, nc=3)

autoencoder = autoencoder.to(device)

criterion = nn.MSELoss(size_average=True)

# Optimizers
optim = optim.Adam(autoencoder.parameters(), lr=args.lr)


for epoch in range(args.epochs):
    step = 0
    for images, _ in tqdm(train_loader):

        images = images.to(device)

        optim.zero_grad()

        batch_size = images.size()[0]

        z_real = autoencoder.encode(images)
        x_recon = autoencoder.decode(z_real)

        recon_loss = criterion(x_recon, images)
        loss_PP_real = plumGauss(z_real)
        loss = recon_loss + args.ld * loss_PP_real

        loss.backward()
        optim.step()


        # step += 1
        # if (step + 1) % 300 == 0:
        #     # print(np.cov(z_real.detach().cpu().numpy()))
        #
        #     print("Epoch: [%d/%d], Step: [%d/%d], Reconstruction Loss: %.4f" %
        #           (epoch + 1, args.epochs, step + 1, len(train_loader), recon_loss.data.item()))


    # noise = torch.randn(batch_size, args.n_z).cuda()
    # samples = autoencoder.decode(noise)



    test(epoch, autoencoder,args.save_dir, test_loader, device, args.batch_size, criterion)

    if not os.path.isdir(args.save_dir + '/saved_models'):
        os.makedirs(args.save_dir + '/saved_models')
    torch.save(autoencoder.state_dict(), '%s/autoencoder_epoch_%d.pth' % (args.save_dir + '/saved_models', epoch))
    # save_image(samples.data, args.save_dir + '/fake_images/images_%d.png' % (epoch + 1))



