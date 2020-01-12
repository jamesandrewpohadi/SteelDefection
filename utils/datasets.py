from torch.utils import data
import pandas as pd
import os
# from PIL import Image
import imageio
import torch
import numpy as np

class SteelDefectDataset(data.Dataset):
    def __init__ (self, csv_file,root_dir, transform=None):
        self.root_dir = root_dir
        self.data = pd.read_csv(os.path.join(root_dir,csv_file))
        self.transform = transform
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        row = self.data.iloc[idx]
        img = imageio.imread(os.path.join(self.root_dir,'train_images',row['ImageId']))
        c = row['ClassId']
        pixel_enc = [int(x) for x in row['EncodedPixels'].split()]
        pixels = []
        for i in range(int(len(pixel_enc)/2)):
            for j in range(pixel_enc[2*i+1]):
                pixels.append(pixel_enc[2*i]-1+j)
        x,y = np.unravel_index(pixels,img.shape[:2][::-1])
        target = torch.zeros(img.shape[:2])
        target[y,x] = 1
        sample = {'img':img,'c':c,'target':target}
        if self.transform:
            sample = self.transform(sample)
        return sample