import torch
from torch import optim
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import pathlib
from torch.utils import data
from siamese_dataset_example import SiamesePairDataset
import PIL.Image as Image
import matplotlib.pyplot as plt
import time
import numpy as np
import torch.nn.functional as F
import csv
import pandas
from sklearn import metrics
import itertools
from models.siamese_resnet import SiameseResnet
import log_compiler
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import random
import PIL
from tqdm import tqdm


trfm_valid = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

trfm_train = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

target_trfrm = transforms.Compose([
    lambda x: [x],
    torch.Tensor
])


# ### Hyper parameters
# A few arbitrary predefined parameters.

# In[100]:


BATCH_SIZE = 256
LEARNING_RATE = 0.003
NUM_EPOCH = 10
ARCHITECTURE = 'resnet101'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRINT_EVERY = 100

# ### Dataset

# In[35]:


train_path = 'dataset/train/facescrub'
valid_path = 'dataset/valid/facescrub'


# #### Siamese Network Dataset

# #### Siamese Pair Dataset

# In[101]:


trainset = SiamesePairDataset(root=train_path, ext='', transform=trfm_train, target_transform=target_trfrm, glob_pattern='*/*.[jJpP]*')
validset = SiamesePairDataset(root=valid_path, ext='', transform=trfm_valid, target_transform=target_trfrm, glob_pattern='*/*.[jJpP]*')
trainloader = data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
validloader = data.DataLoader(validset, batch_size=BATCH_SIZE, shuffle=True)


# In[102]:


print(f'Number of train loader: {len(trainloader)}, number of valid loader ({len(validloader)}), each has {BATCH_SIZE}')


# ### Contrastive Loss
# This function used to calculate the loss/cost of our input image, based on this paper by Yan Lecun http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf. Because siamese is not classification problem rather a distance problem which means we need to compute the difference between two images hence we need another type of loss function.

# In[7]:


class ContrastiveLoss(nn.Module):
    def __init__(self, device, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.device = device

    def forward(self, output1, output2, Y):
        euclidean_distance = F.pairwise_distance(output1, output2)
        euclidean_distance = euclidean_distance.to(self.device)
        loss_contrastive = torch.mean((1-Y) * torch.pow(euclidean_distance, 2) +
                                      (Y) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive


# ### Siamesse Network class, based on Resnet18 model
# We are using resnet18 model with custom fully connected layer based on our very case. But first, we need to freeze layer parameters so they dont get recalculated on the training. The only layer we are training is last layer which is the fully connected layer that has been customized to our case.

# ### Few setups before training

# #### Setup the methods

# In[8]:


def rounding(val, threshold=0.5):
    return val > threshold

def log_training_result(numepoch, batchsize, lrate, accuracy, precision, f1, tp, tn, fp, fn, fc, model, name='training_logs.csv'):
    with open(name, 'r') as f:
        train_number = len(f.readlines())
    detail = log_compiler.compile(numepoch, batchsize, lrate, accuracy, precision, f1, tp, tn, fp, fn, fc, train_number, model)
    data = [batchsize,lrate,numepoch,round(accuracy, 2),round(f1, 2),round(precision, 2),tp,tn,fp,fn,f'logs/{train_number}.md']
    log_compiler.write_log_file(train_number, detail)
    with open(name, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(data)
    print("data saved to %s" % name)
    
def print_scores(acc, err, batch_size):
    # just an utility printing function
    for name, scores in zip(("accuracy", "error"), (acc, err)):
        print(f"\t{name.rjust(14, ' ')}: {sum(scores)/batch_size:.4f}")

def confusion_matrix(y_pred, y_true, threshold=0.5):
    y_pred, y_true = y_pred.view(-1), y_true.view(-1).int()
    y_pred = rounding(y_pred, threshold).int()
#     fn = (1-y_pred && y_true).sum()
#     ap = (y_pred && y_true).sum()
#     an = (1-y_pred && 1-y_true).sum()
#     fp = (y_pred && 1-y_true).sum()
#     fn,fp,ap,an = fn.item(),fp.item(),ap.item(),an.item()
#     error = (fn + fp) / (fp+fn+ap+an) * 100
#     accuracy = (ap+an) / (fp+fn+ap+an) * 100
    corrects = y_true == y_pred
    accuracy = torch.mean(corrects.type(torch.FloatTensor))
    return accuracy, 1-accuracy #accuracy, errors


# In[98]:


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


def train(epoch, num_epoch, model, dataloader, criterion, optimizer):
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    model.train()
    
    end_time = time.time()
    for idx, ((img1,img2),label) in enumerate(dataloader):
        data_time.update(time.time() - end_time)
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)
        optimizer.zero_grad()
        output1, output2 = model.forward(img1, img2) #forward prop
        loss = criterion(output1, output2, label)
        loss.backward()
        optimizer.step()
        
        losses.update(loss.item(), img1.size(0))
        batch_time.update(time.time() - end_time)
        end_time = time.time()
        
        if idx % PRINT_EVERY==0:
            print(f'Train Epoch [{epoch+1}/{num_epoch}] [{idx}/{len(dataloader)}]\t'
                  f' Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f' Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  f' Loss {losses.val:.4f} ({losses.avg:.4f}) ')
        
    return losses.avg
        
def valid(epoch, num_epoch, model, dataloader, criterion, optimizer):
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracy = 0
    accl = []
    
    model.eval()
    with torch.no_grad():
        end_time = time.time()
        for idx, ((img1,img2),label) in enumerate(dataloader):
            data_time.update(time.time() - end_time)
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            output1, output2 = model.forward(img1, img2) #forward prop
            loss = criterion(output1, output2, label)

            predicted_label = F.pairwise_distance(output1, output2)
            acc, err = confusion_matrix(predicted_label, label, threshold=0.3)
            accuracy += acc
            accl.append(acc)
            
            losses.update(loss.item(), img1.size(0))
            batch_time.update(time.time() - end_time)
            end_time = time.time()
            
            if idx % PRINT_EVERY==0:
                print(f'Valid Epoch [{epoch + 1}/{num_epoch}] [{idx}/{len(dataloader)}]\t'
                      f' Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      f' Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      f' Loss {losses.val:.4f} ({losses.avg:.4f})\t Accuracy: {accuracy/PRINT_EVERY:.3f}')
                accuracy = 0
            print(f"Valid accuracy one complete epoch: {sum(accl)/len(accl)}")
            
    return losses.avg


# #### Setup data

# #### Setup the tools

# In[184]:


start_ts = time.time()

#optimizer = optim.Adam([
#    {'params': model.model.conv1.parameters()},
#    {'params': model.model.bn1.parameters()},
#    {'params': model.model.relu.parameters()},
#    {'params': model.model.maxpool.parameters()},
#    {'params': model.model.layer1.parameters(), 'lr': 0.002},
#    {'params': model.model.layer2.parameters(), 'lr': 0.002},
#    {'params': model.model.layer3.parameters(), 'lr': 0.002},
#    {'params': model.model.layer4.parameters(), 'lr': 0.003},
#    {'params': model.model.avgpool.parameters(), 'lr': 0.003},
#    {'params': model.model.fc.parameters(), 'lr': 0.003}
#], lr= 0.001)


# In[161]:

class SiameseTrainer(nn.Module):
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, p, q):
        return self.model(p), self.model(q)

base = models.resnet50(pretrained=True)

for param in base.parameters():
    param.requires_grad = False
    
# for param in b
    
in_features = base.fc.in_features
base.fc = nn.Sequential(
    nn.BatchNorm1d(in_features),
    nn.Linear(in_features, 512),
    nn.ReLU(),
    nn.BatchNorm1d(512),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.BatchNorm1d(256),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 128)
)

criterion = ContrastiveLoss(device)
optimizer = optim.SGD(base.fc.parameters(), lr=.003)

base.to(device)
base = nn.DataParallel(base)
model = SiameseTrainer(base)

print('model created!')


# In[108]:

#TODO: increase learning rate here
#optimizer = optim.Adam(base.module.fc.parameters(), lr=.005)

train_losses, valid_losses = [], []

print("Start training", time.strftime('%X %x %Z'))
time_start = time.time()

for epoch in range(NUM_EPOCH):
    trainloss = train(epoch, NUM_EPOCH, model, trainloader, criterion, optimizer)
    validloss = valid(epoch, NUM_EPOCH, model, validloader, criterion, optimizer)
    train_losses.append(trainloss)
    valid_losses.append(validloss)

print("End training", time.strftime('%X %x %Z'))
print(f"Training finished in {time.time()-time_start}")

plt.plot(train_losses, label='train_loss')
plt.plot(valid_losses, label='valid_loss')
plt.legend(frameon=False)
plt.savefig('facescrub-training.jpg')
print("plot saved to facescrub-training.jpg")


model.eval()
progress = tqdm(enumerate(validloader, 1))
len_valid = len(validloader)
laccuracy = []
with torch.no_grad():
    for idx, ((img1,img2),label) in progress:
        img1 = img1.to(device)
        img2 = img2.to(device)
        label = label.to(device)
        out1, out2 = model.forward(img1,img2)
        predicted_label = F.pairwise_distance(out1, out2)
        
        
        acc, err = confusion_matrix(predicted_label, label, threshold=0.3)
        progress.set_description(f'Val acc {idx}/{len_valid}: {acc.item():.3f}')
        laccuracy.append(acc.item())

print(f"Avg valid accuracy: {sum(laccuracy)/len(laccuracy)}")
print()
print(f"saving model")
torch.save(model, f"model-resnet-50-acc-{sum(laccuracy)/len(laccuracy):.2f}.pth")
print()
print("--- END ---")

