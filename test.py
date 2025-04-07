import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader
from torchvision.transforms import ToTensor
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

import os
classname = os.listdir('/Users/khevinjugessur/Documents/ENEL525/Project/UCMerced_LandUse/Images')
path_to_image = '/Users/khevinjugessur/Documents/ENEL525/Project/UCMerced_LandUse/Images'
img_paths = []
classes_img = []
for cls in classname:
    path_to_cls_img = os.listdir(f'/Users/khevinjugessur/Documents/ENEL525/Project/UCMerced_LandUse/Images/{cls}')
    for sub in path_to_cls_img:
        img_paths.append(f'/Users/khevinjugessur/Documents/ENEL525/Project/UCMerced_LandUse/Images/{cls}/{sub}')
        classes_img.append(cls)
        
        
from sklearn.model_selection import train_test_split
import numpy as np

from sklearn.preprocessing import OneHotEncoder,LabelEncoder
labelencode = LabelEncoder()
labels = labelencode.fit_transform(classes_img)
# labels = labels.reshape(-1,1)
# onehotencode = OneHotEncoder(drop='first')
# encoded_class = onehotencode.fit_transform(labels)

X_train,X_test,y_train,y_test = train_test_split(img_paths,labels,test_size = 0.2,random_state = 10)

Transform_pipline = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor()
])


class Dataset_class(Dataset):
    def __init__(self,paths,labels,transform):
        super().__init__()
        self.paths = paths
        self.transform=transform
        self.labels = labels
        self.length = labels.shape[0]
        
    def __len__(self):
        return self.length
    
    def __getitem__(self,idx):
#         print("This is getitem idx :",idx,type(idx))
        img_path = self.paths[idx]
        img = Image.open(img_path)
        img = self.transform(img)
#         print(type(img),img.shape)
        label = self.labels[idx]
        return img,label
    
    
    
train_dataset = Dataset_class(X_train,y_train,Transform_pipline)
test_dataset = Dataset_class(X_test,y_test,Transform_pipline)

batch_size = 16
Train_DL= DataLoader(
    dataset = train_dataset,
    shuffle = True,
    batch_size = batch_size
)

Test_DL = DataLoader(
    dataset = test_dataset,
    shuffle = True,
    batch_size = batch_size
)

class UCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu=nn.ReLU()
        self.drop = nn.Dropout(0.5)
        self.conv1=nn.Conv2d(in_channels=3,out_channels=8, kernel_size=(7,7), stride=3, padding=0)
        self.bn1=nn.BatchNorm2d(8)
        self.mp1=nn.AvgPool2d(kernel_size=(2,2),stride=2,padding=0)
        
        self.conv2=nn.Conv2d(in_channels=8,out_channels=16, kernel_size=(3,3), stride=1, padding=1)
        self.bn2=nn.BatchNorm2d(16)
        
        self.conv3=nn.Conv2d(in_channels=16,out_channels=32, kernel_size=(3,3), stride=1, padding=1)
        self.bn3=nn.BatchNorm2d(32)

        self.conv4=nn.Conv2d(in_channels=32,out_channels=64, kernel_size=(3,3), stride=1, padding=1)
        self.bn4=nn.BatchNorm2d(64)
        
        self.mp2=nn.AvgPool2d(kernel_size=(2,2),stride=2,padding=0)
        
        self.conv5=nn.Conv2d(in_channels=64,out_channels=128, kernel_size=(3,3), stride=2, padding=0)
        self.bn5=nn.BatchNorm2d(128)
        
        
        self.flatten=nn.Flatten()

        self.lin1=nn.Linear(in_features=12800, out_features=64)
        self.lin2 = nn.Linear(in_features = 64, out_features = 21)

        self.softmax = nn.Softmax(dim=1)
    
    def forward(self,x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.mp1(x)
        
        
        x=self.conv2(x)
        x=self.bn2(x)
        x=self.relu(x)
        
        x=self.conv3(x)
        x=self.bn3(x)
        x=self.relu(x)
        
        x=self.conv4(x)
        x=self.bn4(x)
        x=self.relu(x)
        x=self.mp2(x)
        
        x=self.conv5(x)
        x=self.bn5(x)
        x=self.relu(x)
        
        x=self.flatten(x)
        x=self.drop(x)
        x=self.lin1(x)
        x=self.drop(x)
        x=self.lin2(x)
#         x=self.drop(x)
        probs = self.softmax(x)
        return x      
    
model=UCNN()

loss_fn=nn.CrossEntropyLoss()
lr=0.001
optimizer=torch.optim.Adam(params=model.parameters(), lr=lr)
n_epochs=20


def train_one_epoch(dataloader, model,loss_fn, optimizer):
    model.train()
    track_loss=0
    num_correct=0
    for i, (imgs, labels) in enumerate(dataloader):
        
        pred=model(imgs)
#         print(pred.shape,labels.shape)           
        loss=loss_fn(pred,labels)
        track_loss+=loss.item()
        num_correct+=(torch.argmax(pred,dim=1)==labels).type(torch.float).sum().item()
        
        running_loss=round(track_loss/(i+(imgs.shape[0]/batch_size)),2)
        running_acc=round((num_correct/((i*batch_size+imgs.shape[0])))*100,2)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i%100==0:
            print("Batch:", i+1, "/",len(dataloader), "Running Loss:",running_loss, "Running Accuracy:",running_acc)
            
    epoch_loss=running_loss
    epoch_acc=running_acc
    return epoch_loss, epoch_acc


def test_one_epoch(dataloader, model):
    model.train()
    track_loss=0
    num_correct=0
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(dataloader):
            pred=model(imgs)
            loss=loss_fn(pred,labels)
            track_loss+=loss.item()
            num_correct+=(torch.argmax(pred,dim=1)==labels).type(torch.float).sum().item()

            running_loss=round(track_loss/(i+(imgs.shape[0]/batch_size)),2)
            running_acc=round((num_correct/((i*batch_size+imgs.shape[0])))*100,2)
        
    epoch_loss=running_loss
    epoch_acc=running_acc
    return epoch_loss, epoch_acc


for i in range(n_epochs):
    print("Epoch No:",i+1)
    train_epoch_loss, train_epoch_acc=train_one_epoch(Train_DL,model,loss_fn,optimizer)
    print("Training:", "Epoch Loss:", train_epoch_loss, "Epoch Accuracy:", train_epoch_acc)
    print("--------------------------------------------------")