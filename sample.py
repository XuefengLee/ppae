import argparse
import torch
import os, math
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.datasets import CelebA
from torchvision.transforms import transforms
from torchvision.utils import save_image
from model import celeba_model, vae_model
import pdb

torch.manual_seed(123)

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--use_cuda', type=bool, default=True)
parser.add_argument('--model_path', required=True, help='path to the model')
parser.add_argument('--save_dir', required=True, help='path to save dir')
parser.add_argument('--device', type=int,  default=0, help='cuda device')
parser.add_argument('--test', type=bool, default=False)


opt = parser.parse_args()


device = opt.device





celebTrans = transforms.Compose([
    transforms.CenterCrop(140),
    transforms.Resize(64),
    transforms.ToTensor()
])

trainset = CelebA(root=opt.dataroot,
                 split='train',
                 transform=celebTrans,
                 download=True)

train_loader = DataLoader(dataset=trainset,
                         batch_size=100,
                         shuffle=False)


testset = CelebA(root=opt.dataroot,
                 split='test',
                 transform=celebTrans,
                 download=True)

test_loader = DataLoader(dataset=testset,
                         batch_size=1,
                         shuffle=False)


if __name__ == "__main__":


    if opt.test:
        if not os.path.isdir('./data/test_samples'):
            os.makedirs('./data/test_samples')
        i = 0
        for images, _ in tqdm(test_loader):
            save_image(images.data, './data/test_samples/wae_gan_images_%d.png' % (i))
            print("%d images saved" % (i))
            i += 1
            if i == 10000:
                break


    else:

        model_state_dict = torch.load(opt.model_path,map_location='cuda:' + str(device))
        model = celeba_model()
        # model = vae_model()
        model.load_state_dict(model_state_dict)

        model = model.to(device)
        model.eval()
        full_z = []

        i  = 0
        with torch.no_grad():
            for i, (data, _) in enumerate(train_loader):
                data = data.to(device)

                z_real = model.encode(data)

                full_z.append(z_real)

                if i + 1 == 100:
                    break

        full_z = torch.cat(full_z, dim=0).detach().cpu().numpy()
        np.savetxt('cov3.txt',np.cov(full_z.T))
        # # pdb.set_trace()
        cov = np.loadtxt('cov3.txt')
        # norms = [8 - math.sqrt(2), 8 - math.sqrt(0.5), 8, 8 + math.sqrt(2), 8 + math.sqrt(0.5)]

        # for index, new_norm in enumerate(norms):
        index = 1
        dir = opt.save_dir + '/samples/' + str(index)
        if not os.path.isdir(dir):
            os.makedirs(dir)

        E, U = torch.symeig(torch.from_numpy(cov).float(),eigenvectors=True)
        UE = torch.mm(U, E.diag())
        np.savetxt('eig3.txt',E)

        for i in range(2050):

            j = i % 41
            if j == 0:
                print(i)
                standard_noise = torch.randn(64,1)
            standard_noise[59] = -2 + 0.1 * j
            # noise = np.random.multivariate_normal([0]*64, cov)
            # noise = torch.from_numpy(noise).to(device).float()
            noise = torch.mm(UE,standard_noise).transpose(1,0).cuda()

            # noise = torch.randn(1, 64).cuda()
            #
            # old_norm = torch.norm(noise, 1)
            # noise = noise * ((8 - math.sqrt(2))/ old_norm)

            samples = model.decode(noise)

            save_image(samples.data, dir +  '/images_%d.png' % (i))
            print("%d images saved" %(i))


