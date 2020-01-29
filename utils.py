import torch
import math, os, pdb
from torchvision.utils import save_image


def tocuda(x):

    return x.cuda()

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.bias.data.fill_(0)


def log_sum_exp(input):
    m, _ = torch.max(input, dim=1, keepdim=True)
    input0 = input - m
    m.squeeze()
    return m + torch.log(torch.sum(torch.exp(input0), dim=1))


def get_log_odds(raw_marginals):
    marginals = torch.clamp(raw_marginals.mean(dim=0), 1e-7, 1 - 1e-7)
    return torch.log(marginals / (1 - marginals))



def plumGauss(z):

    x0 = torch.squeeze(z)
    x1 = x0.transpose(0,1)

    batch = x0.size()[0]
    dim = x0.size()[1]  # vector length

    xx  = torch.bmm(x0.view(batch, 1, dim),
                    x0.view(batch, dim, 1)).squeeze(2)
    xx0 = xx.expand(batch,batch)
    xx1 = xx0.transpose(0,1)

    xy = xx0 + xx1 - 2*torch.matmul(x0,x1)

    xx2 = 2*xx*(torch.log(1+4*xx/dim)+1.3-math.log(5))/(1+1.6*xx/dim)

    result = 0.55*torch.sum(xx)/(1+2*dim) + 0.45*torch.sum(xx2)/(1+2*dim) \
           - 0.5*torch.sum(torch.log(1+xy*2/(1+2*dim)))/(batch-1)
    return math.sqrt(1+2*dim)*result/batch

def test(epoch, model,save_dir,test_loader, device, batch_size,criterion):
    model.eval()
    test_loss = 0

    if not os.path.isdir(save_dir+'/images'):
        os.makedirs(save_dir+'/images')

    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)

            z_real = model.encode(data)
            recon_batch = model.decode(z_real)

            # calculate mean and variance
            mean = torch.norm(z_real, dim=1).pow(2).mean() / z_real.shape[1]
            var = (torch.norm(z_real, dim=1) - z_real.shape[1]).pow(2).mean()/(2*z_real.shape[1])

            test_loss += criterion(recon_batch, data).item()

            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],recon_batch.view(batch_size, 1, 28, 28)[:n]])
                # comparison = torch.cat([data[:n],recon_batch.view(batch_size, 3, 64, 64)[:n]])

                save_image(comparison.cpu(), save_dir + '/images/reconstruction_' + str(epoch) + '.png', nrow=n)


    test_loss /= len(test_loader.dataset)
    print("Epoch: %d, Reconstruction Loss: %.4f, Mean: %.4f, Variance: %.4f" %
          (epoch + 1, test_loss, mean.data.item(), var.data.item()),flush=True)