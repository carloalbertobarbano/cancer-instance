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
        mask = np.load(mask).astype('float').transpose(2, 0, 1) #256x256x6
    
        t = self.T(image=image, mask=mask)
        return t['image'], t['mask']

class ChannelDiceLoss(smp.utils.base.Loss):
    def __init__(self, eps=1., beta=1., activation=None, **kwargs):
        super().__init__(**kwargs)
        self.dice_loss = smp.utils.losses.DiceLoss(eps=eps, beta=beta, activation=activation)
        
    def forward(self, y_pr, y_gt):
        loss = []

        for i in range(y_pr.shape[1]):
            channel_loss = self.dice_loss(y_pr[:, i][:, None], y_gt[:, i][:, None])
            loss.append(channel_loss)
        
        return torch.stack(loss, dim=0).mean()

class AMPTrainEpoch(smp.utils.train.TrainEpoch):
    def __init__(self, model, loss, metrics, optimizer, device='cpu', verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='train',
            device=device,
            verbose=verbose,
        )

        self.scaler = torch.cuda.amp.GradScaler()

    def batch_update(self, x, y):
        self.optimizer.zero_grad()

        with torch.cuda.amp.autocast(): 
            prediction = self.model.forward(x)
            loss = self.loss(prediction, y)
        
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss, prediction
    

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
        classes=6,
        activation='softmax'
    ).to(config.device)

    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, weight_decay=config.weight_decay, momentum=config.momentum)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = ChannelDiceLoss()

    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]
    
    trainer = AMPTrainEpoch if config.amp else smp.utils.train.TrainEpoch
    train_epoch = trainer(
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

    color_mapping = torch.tensor([[1., 0, 0], #neoplastic
                                  [0, 1., 0], #infl
                                  [0, 0, 1.], #connective/soft
                                  [1., 1., 0], #dead
                                  [0, 1., 1.], #ephitelia
                                  [0, 0, 0]]) #background

    test_batch = next(iter(test_loader))
    test_batch_im, test_batch_gt = test_batch 
    test_batch_im = torchvision.utils.make_grid(test_batch_im)
    test_batch_gt = torch.argmax(test_batch_gt, dim=1)
    test_batch_gt = torch.cat([color_mapping[test_batch_gt[i]] for i in range(test_batch_gt.shape[0])], dim=0)
    test_batch_gt = torchvision.utils.make_grid(test_batch_gt, normalize=False).numpy().astype('float')

    wandb.log({
        'test_batch': wandb.Image(test_batch_im),
        'test_batch_gt': wandb.Image(test_batch_gt)
    }, commit=False)
    
    for epoch in range(config.epochs):
        print(f'-------- Epoch {epoch+1} --------')
        train = train_epoch.run(train_loader)
        valid = valid_epoch.run(valid_loader)
        test = valid_epoch.run(test_loader)

        scheduler.step()
        
        wandb.log({
            'train': train,
            'valid': valid,
            'test': test,
            'iou': valid['iou_score']
        }, commit=False)

        with torch.no_grad():
            masks = model(test_batch[0].to(config.device)).cpu()
            masks = torch.argmax(masks, dim=1)
            masks = torch.cat([color_mapping[masks[i]] for i in range(masks.shape[0])], dim=0)
        masks = torchvision.utils.make_grid(masks, normalize=False).numpy().astype('float')

        wandb.log({
            'masks': wandb.Image(masks)
        })

        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'config': config}, os.path.join(wandb.run.dir, 'model.pth'))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42) 
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--root', type=str, default=f'/data/cancer-instance')

    parser.add_argument('--amp', action='store_true', help='Enable mixed precision training')

    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--momentum', type=float, default=0.9)

    parser.set_defaults(amp=False)
    config = parser.parse_args()

    wandb.init(project='cancer-instance')

    main(config)

    