import torch
from torchvision import transforms
from tqdm import tqdm
import utils
from torch.utils.data import DataLoader
from models.Unet import UNet
import os

# hyperparams
epoch = 4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(n_channels=3, n_classes=4, bilinear=True)
print(model)
model.to(device)

optimizer = torch.optim.SGD(model.parameters(),lr=0.001,momentum=0.9)

transform = transforms.Compose([
    utils.transforms.RandomMirror(),
    utils.transforms.ToTensor(),
    utils.transforms.Downsize(2)
])

dataset = utils.datasets.SteelDefectDataset(csv_file='train.csv',
root_dir='data/severstal-steel-defect-detection',transform=transform)
train_loader = DataLoader(dataset, batch_size=1,shuffle=True)

criterion = utils.loss.SegmentMSELoss()

for e in range(1,epoch+1):
    print('Epoch {}:'.format(e))
    total_loss = 0
    for batch, data in tqdm(enumerate(train_loader),total=len(train_loader),leave=False):
        optimizer.zero_grad()
        imgs, cs, targets = data['img'], data['c'], data['target']
        imgs = imgs.to(device)
        targets = targets.to(device)
        out = model(imgs)
        loss = criterion(out,cs,targets)
        loss.backward()
        total_loss += loss.data
        optimizer.step()
        if batch == 500:
            print(total_loss/batch)
    print('Loss: {:.3f}'.format(total_loss/(batch)))
    torch.save(model.state_dict(), os.path.join('weights','Unet_e{}.pth'.format(e)))
