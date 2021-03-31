import torch
import torchvision
import cv2
import albumentations
import albumentations.pytorch
import numpy as np
import segmentation_models_pytorch as smp

from tqdm import tqdm
from torch.utils import data

import wandb
import argparse
import os
import random
import glob

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)

class CancerInstance(data.Dataset):
    def __init__(self, root, T):
        self.root = root   
        self.T = T

        self.images = sorted(glob.glob(os.path.join(root, 'data', '*.png')))
        self.masks = sorted(glob.glob(os.path.join(root, 'masks', '*.npy')))

        print(f'=> Loaded {len(self.images)} images')

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image, mask = self.images[index], self.masks[index]

        image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB) #256x256x3
        mask = np.load(mask) #256x256x6

        t = self.T(image=image, mask=mask)
        return t['image'], t['mask']

class DiceScore(torch.nn.Module):
    __name__ = 'dice_score'

    def forward(self, input, target):
        smooth = 1.0
        iflat = input.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()
        return ((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))

def run(model, dataloader, criterion, optimizer, device):
    train = optimizer is not None
    tot_loss = 0.

    model.train(train)
    for data, masks in tqdm(dataloader, leave=False):
        data, masks = data.to(device), masks.to(device)

        with torch.set_grad_enabled(train):
            output = model(data)


def main(config):
    set_seed(config.seed)

    print('=> Loading data')
    T = albumentations.Compose([
        albumentations.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        albumentations.pytorch.ToTensorV2()
    ])

    dataset = CancerInstance(root=config.root, T=T)
    train_dataset, valid_dataset, test_dataset = data.random_split(dataset, [4425, 1106, 2370]) #0.7*0.8, 0.7*0.2, 0.3

    train_loader = data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = data.DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print('=> Loading model')
    model = smp.Unet(
        encoder_name="resnet18",        
        encoder_weights="imagenet",     
        in_channels=3,                 
        classes=5,
        activation='sigmoid'
    ).to(config.device)

    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = smp.utils.losses.DiceLoss()

    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]
    
    train_epoch = smp.utils.train.TrainEpoch(
        model, 
        loss=criterion, 
        metrics=metrics, 
        optimizer=optimizer,
        device=config.device,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model, 
        loss=criterion, 
        metrics=metrics, 
        device=config.device,
        verbose=True,
    )

    color_mapping = torch.tensor([[255, 0, 0], #neoplastic
                                  [0, 255, 0], #infl
                                  [0, 0, 255], #connective/soft
                                  [255, 255, 0], #dead
                                  [0, 255, 255], #ephitelia
                                  [0, 0, 0]]) #background

    test_batch = next(iter(test_loader))
    test_batch_im = torchvision.utils.make_grid(test_batch)

    wandb.log({
        'test_batch': wandb.Image(test_batch_im)
    }, commit=False)
    
    for epoch in range(config.epochs):
        print(f'-------- Epoch {epoch+1} --------')
        train = train_epoch(train_loader)
        valid = valid_epoch(valid_loader)
        test = valid_epoch(test_loader)

        scheduler.step()
        
        wandb.log({
            'train': train,
            'valid': valid,
            'test': test
        }, commit=False)

        with torch.no_grad():
            masks = model(test_batch)
            masks = torch.argmax(outputs, dim=1)
            masks = torch.cat([color_mapping[m] for m in masks], dim=0)
        masks = torchvision.utils.make_grid(masks)

        wandb.log({
            'masks': wandb.Image(masks)
        })

        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'config': config}, os.path.join(wandb.run.dir, 'model.pth'))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42) 
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--root', type=str, default=f'{os.path.expanduser("~")}/data/cancer-instance')


    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)

    config = parser.parse_args()

    wandb.init(
        project='cancer-instance'
    )

    main(config)

    