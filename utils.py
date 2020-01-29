import torch
import math


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

    batch_size = x0.size()[0]
    size = x0.size()[1]  # vector length

    xx  = torch.bmm(x0.view(batch_size, 1, size),
                    x0.view(batch_size, size, 1)).squeeze(2)
    xx0 = xx.expand(batch_size,batch_size)
    xx1 = xx0.transpose(0,1)

    xy = xx0 + xx1 - 2*torch.matmul(x0,x1)

    #result = torch.sum(xx)/size - 2*torch.sum(torch.log(1+xy/(2*size)))/batch_size
    #return math.sqrt(size)*result/(batch_size-1)

    result = 4*torch.sum(xx/(1+torch.log(1+0.177*xx/size)))/(3*(0.5+2*size)) \
             - torch.sum(torch.log(1+xy/(0.5+2*size)))/(batch_size-1)

    return math.sqrt(0.5+2*size)*result/batch_size