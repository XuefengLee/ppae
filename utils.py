import torch
import math, os, pdb
from torchvision.utils import save_image
import numpy as np

def plumGauss(z, alpha=0.45):

    x0 = torch.squeeze(z)
    x1 = x0.transpose(0,1)

    batch = x0.size()[0]
    dim = x0.size()[1]  # vector length

    xx  = torch.bmm(x0.view(batch, 1, dim),
                    x0.view(batch, dim, 1)).squeeze(2)
    xx0 = xx.expand(batch,batch)
    xx1 = xx0.transpose(0,1)

    xy = xx0 + xx1 - 2*torch.matmul(x0,x1)

    if alpha == 0:
        result = torch.sum(xx)/(1+2*dim) - 0.5*torch.sum(torch.log(1+xy*2/(1+2*dim)))/(batch-1)
    else:
        xx2 = 2*xx*(torch.log(1+4*xx/dim)+1.3-math.log(5))/(1+1.6*xx/dim)

    result = 0.55*torch.sum(xx)/(1+2*dim) + 0.45*torch.sum(xx2)/(1+2*dim) \
           - 0.5*torch.sum(torch.log(1+xy*2/(1+2*dim)))/(batch-1)

    return math.sqrt(1+2*dim)*result/batch

def test(epoch, model,save_dir,test_loader, device, batch_size, criterion, scale):
    model.eval()
    test_loss = 0

    if not os.path.isdir(save_dir+'/images'):
        os.makedirs(save_dir+'/images')

    full_z = []
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)

            z_real = model.encode(data)

            full_z.append(z_real)

            recon_batch = model.decode(z_real)

            test_loss += criterion(recon_batch, data).item()

            # if i == 0:

        full_z = scale*torch.cat(full_z,dim=0)


        n = min(data.size(0), 8)
        # comparison = torch.cat([data[:n],recon_batch.view(batch_size, 1, 28, 28)[:n]])
        comparison = torch.cat([data[:n],recon_batch.view(data.shape[0], 3, 64, 64)[:n]])

        E1 = torch.norm(full_z, dim=1).pow(2).mean()
        E2 =(torch.norm(full_z, dim=1).pow(2) - E1).pow(2).mean()
        mean = E1/full_z.shape[1]
        var = E2/(2*E1)
        #mean = torch.norm(z_real, dim=1).pow(2).mean() / z_real.shape[1]
        #var = (torch.norm(z_real, dim=1).pow(2) - mean*z_real.shape[1]).pow(2).mean() / (2 * z_real.shape[1])
        cov = np.cov(full_z.detach().cpu().numpy())

        print(cov)
        save_image(comparison.cpu(), save_dir + '/images/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader)
    print("Epoch: %d, Reconstruction Loss: %.4f, Mean: %.4f, Variance: %.4f" %
          (epoch + 1, test_loss, mean.data.item(), var.data.item()),flush=True)
    model.train()

    return mean.data.item()
