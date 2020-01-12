import torch.nn.functional as F
from torchvision import transforms
from random import random

class Downsize(object):
    def __init__(self,scale):
        self.scale = scale

    def __call__(self,sample):
        img, c, target = sample['img'], sample['c'], sample['target']
        img = F.interpolate(img.unsqueeze(0),scale_factor=1/self.scale).squeeze(0)
        target = F.interpolate(target.unsqueeze(0).unsqueeze(0),scale_factor=1/self.scale).squeeze(0).squeeze(0)
        return {'img':img,'c':c,'target':target}

class RandomMirror(object):
    def __call__(self,sample):
        if random() > 0.5:
            return sample
        else:
            img, c, target = sample['img'], sample['c'], sample['target']
            assert img.shape[1] == target.shape[1]
            n = img.shape[1]
            flip = range(n)[::-1]
            return {'img':img[:,flip],'c':c,'target':target[:,flip]}

class ToTensor(object):
    def __call__(self,sample):
        img, c, target = sample['img'], sample['c'], sample['target']
        return {'img':transforms.ToTensor()(img),'c':c,'target':target}