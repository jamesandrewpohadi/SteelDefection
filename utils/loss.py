import torch.nn as nn
import torch

class SegmentMSELoss(nn.Module):
    def __init__(self):
        super(SegmentMSELoss, self).__init__()

    def forward(self,output,c,target):
        # device = torch.device('cuda' if output.is_cuda else 'cpu')
        m,h,w = target.shape
        output = output[:,c-1,:,:]
        mse = (target-output)**2
        mask0 = torch.ones(target.shape).cuda()-target
        mask1 = target
        mean = target.mean()
        c0 = 1/(1-mean)
        c1 = 1/mean
        loss = mse*(c0*mask0+c1*mask1)
        loss = loss.mean()
        return loss
