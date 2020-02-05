import argparse
import torch
import os, math
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.datasets import CelebA
from torchvision.transforms import transforms
from torchvision.utils import save_image
from model import celeba_model
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
        model.load_state_dict(model_state_dict)

        model = model.to(device)

        norms = [8 - math.sqrt(2), 8 - math.sqrt(0.5), 8, 8 + math.sqrt(2), 8 + math.sqrt(0.5)]

        for index, new_norm in enumerate(norms):

            dir = opt.save_dir + '/samples/' + str(index)
            if not os.path.isdir(dir):
                os.makedirs(dir)
            for i in range(10000):

                noise = torch.randn(1, 64).cuda()
                old_norm = torch.norm(noise, 1)
                noise = noise * (new_norm / old_norm)

                # noise = torch.randn(1, 64).cuda()
                samples = model.decode(noise)
                save_image(samples.data, dir +  '/images_%d.png' % (i))
                print("%d images saved" %(i))


