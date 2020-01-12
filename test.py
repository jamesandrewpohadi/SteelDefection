import torch
from torchvision import transforms
from tqdm import tqdm
import utils
from torch.utils.data import DataLoader
from models.Unet import UNet
import os
import matplotlib.pyplot as plt

# hyperparams

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(n_channels=3, n_classes=4, bilinear=True)
print(model)
model.load_state_dict(torch.load('weights/Unet_e3.pth'))
model.to(device)

transform = transforms.Compose([
    utils.transforms.RandomMirror(),
    utils.transforms.ToTensor(),
    utils.transforms.Downsize(2)
])

dataset = utils.datasets.SteelDefectDataset(csv_file='train.csv',
root_dir='data/severstal-steel-defect-detection',transform=transform)
test_loader = DataLoader(dataset, batch_size=1,shuffle=True)

for batch, data in tqdm(enumerate(test_loader),total=len(test_loader),leave=False):
    if batch == 1:
        break
    imgs, cs, targets = data['img'], data['c'], data['target']
    imgs = imgs.to(device)
    targets = targets.to(device)
    out = model(imgs)
    print(cs)
    plt.imshow(targets[0].cpu().numpy())
    plt.show()
    plt.imshow(out[:,[0]][0,0].cpu().detach().numpy())
    plt.show()