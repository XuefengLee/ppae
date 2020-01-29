import torch
import math

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

# test to see that it works
z = torch.FloatTensor([[1,1,-1,1],[1,-1,1,1],[-1,-1,1,1]])
loss = plumGauss(z)
print(loss)
