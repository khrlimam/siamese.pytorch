#!/usr/bin/env python
# coding: utf-8

# In[12]:


import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch import optim
from torch.utils import data

from siamese_dataset_example import SiamesePairDataset


# #### Few classes

# In[3]:


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# ### Contrastive Loss
# This function used to calculate the loss/cost of our input image,
# based on this paper by Yan Lecun http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf.
# Because siamese is not classification problem rather a distance problem
# which means we need to compute the difference between two images hence we need another type of loss function.

# In[4]:


class ContrastiveLoss(nn.Module):
    def __init__(self, device, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.device = device

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        euclidean_distance = euclidean_distance.to(self.device)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


# ### Siamesse Network class, based on Resnet18 model
# We are using resnet18 model with custom fully connected layer based on our very case.
# But first, we need to freeze layer parameters so they dont get recalculated on the training.
# The layer we are training is only the last layer which is the fully connected layer that has been customized to our case.

# In[5]:


class SiameseResnet(nn.Module):
    def __init__(self):
        super(SiameseResnet, self).__init__()
        self.model = models.resnet18(pretrained=True)
        # freezing the parameters so they dont get messed during backprops
        for param in self.model.parameters():
            param.required_grad = False

        # setup our fully connected layer
        in_features = self.model.fc.in_features  # 512
        self.model.fc = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),

            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),

            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),

            nn.BatchNorm1d(64),
            nn.Linear(64, 32)
        )

    def forward(self, img1, img2):
        o_img1 = self.model.forward(img1)
        o_img2 = self.model.forward(img2)
        return o_img1, o_img2


# In[6]:


# x1 = torch.rand(10,3,224,224)
# x2 = torch.rand(10,3,224,224)
# siamese = SiameseResnet()
# o1, o2 = siamese.forward(x1, x2)
# import torch.nn.functional as F
# F.pairwise_distance(o1, o2)


# ### Hyper parameters
# A few arbitrary predefined parameters.

# In[7]:


BATCH_SIZE = 64
LEARNING_RATE = 0.003
PRINT_EVERY = 10
NUM_EPOCH = 10

# ### Setup data

# In[8]:


train_path = 'dataset/train/'
valid_path = 'dataset/valid/'

trfrm = transforms.Compose([
    lambda img: img.convert("RGB"),
    transforms.Resize((224, 224)),
    transforms.CenterCrop((224)),
    transforms.ToTensor(),
])

train_set = SiamesePairDataset(root=train_path, ext='pgm', transform=trfrm)
valid_set = SiamesePairDataset(root=valid_path, ext='pgm', transform=trfrm)
train_loader = data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = data.DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=True)


# ### Few setups before training

# #### Setup the methods

# In[9]:


def train(epoch, num_epoch, model, dataloader, criterion, optimizer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()

    end_time = time.time()
    for idx, (img1, img2, label) in enumerate(dataloader):
        data_time.update(time.time() - end_time)

        img1 = img1.to(device)
        img2 = img2.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        output1, output2 = model.forward(img1, img2)  # forward prop
        loss = criterion(output1, output2, label)
        loss.backward()  # backprop
        optimizer.step()

        losses.update(loss.item(), img1.size(0))
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if idx % PRINT_EVERY == 0:
            print(f'Train Epoch [{epoch + 1}/{num_epoch}] [{idx}/{len(dataloader)}]\t'
                  f' Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f' Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  f' Loss {losses.val:.4f} ({losses.avg:.4f}) ')

    return losses.avg


def valid(epoch, num_epoch, model, dataloader, criterion, optimizer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.eval()
    with torch.no_grad():
        end_time = time.time()
        for idx, (img1, img2, label) in enumerate(dataloader):
            data_time.update(time.time() - end_time)

            img1 = img1.to(device)
            img2 = img2.to(device)
            label = label.to(device)

            output1, output2 = model.forward(img1, img2)  # forward prop
            loss = criterion(output1, output2, label)

            losses.update(loss.item(), img1.size(0))
            batch_time.update(time.time() - end_time)
            end_time = time.time()

            if idx % PRINT_EVERY == 0:
                print(f'Valid Epoch [{epoch + 1}/{num_epoch}] [{idx}/{len(dataloader)}]\t'
                      f' Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      f' Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      f' Loss {losses.val:.4f} ({losses.avg:.4f}) ')

    return losses.avg


# #### Setup the tools

# In[10]:


model = SiameseResnet()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = ContrastiveLoss(device=device)
optimizer = optim.Adam(model.model.fc.parameters(), lr=LEARNING_RATE)

# ### Doing the actual training

# In[15]:


model.to(device)
history = {'train': [], 'valid': []}
for epoch in range(NUM_EPOCH):
    trainloss = train(epoch, NUM_EPOCH, model, train_loader, criterion, optimizer)
    validloss = valid(epoch, NUM_EPOCH, model, valid_loader, criterion, optimizer)
    history['train'].append(trainloss)
    history['valid'].append(validloss)

torch.save(model, 'siamese.pth')
