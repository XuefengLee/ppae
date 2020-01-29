import argparse
import torch
import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.datasets import CelebA
from torchvision.transforms import transforms
from torchvision.utils import save_image
from model import cnn,mlp
import pdb

torch.manual_seed(123)

batch_size = 100
latent_size = 256
cuda_device = "1"

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--use_cuda', type=bool, default=True)
parser.add_argument('--model_path')
parser.add_argument('--device', type=int,  default=0, help='cuda device')
parser.add_argument('--test', type=bool, default=False)



opt = parser.parse_args()


cuda_device = opt.device

os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)

def tocuda(x):
    if opt.use_cuda:
        return x.cuda()
    return x



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

        decoder_state_dict = torch.load(opt.model_path)
        decoder = Decoder()
        decoder.load_state_dict(decoder_state_dict)





        if torch.cuda.is_available():
            decoder = decoder.cuda()

        if not os.path.isdir('./data/samples_2'):
            os.makedirs('./data/samples_2')
        for i in range(10000):
            noise = torch.randn(1, 64).cuda()
            samples = decoder(noise)
            save_image(samples.data, './data/samples_2/wae_gan_images_%d.png' % (i))
            print("%d images saved" %(i))
