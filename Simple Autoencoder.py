#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import scipy.io as sio
import numpy as np
#from pandas import Series
from time import time
import collections
import torch.nn.functional as F
# for creating validation set
from sklearn.model_selection import train_test_split
from tqdm import tqdm, tqdm_notebook
from tqdm import tqdm
import torch.nn as nn
from torch.autograd import Variable as V
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
#import torchvision
from torchvision import transforms
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv1d, MaxPool1d, Module, Softmax, BatchNorm1d, Dropout
from torch.optim import Adam, SGD

# Your Dataset - train and validation
trn_x,val_x,trn_y,val_y = train_test_split(X,y,test_size=0.30, random_state=3)

#Converting to tensor 
trn_x_torch = torch.from_numpy(trn_x).type(torch.FloatTensor)
trn_y_torch = torch.from_numpy(trn_y)

val_x_torch = torch.from_numpy(val_x).type(torch.FloatTensor)
val_y_torch = torch.from_numpy(val_y)

trn = TensorDataset(trn_x_torch,trn_y_torch)
val = TensorDataset(val_x_torch,val_y_torch)

trn_dataloader = torch.utils.data.DataLoader(trn,batch_size=100,shuffle=False, num_workers=4)
val_dataloader = torch.utils.data.DataLoader(val,batch_size=100,shuffle=False, num_workers=4)

#Defining Autoencoder
class AutoEncoder(nn.Module):    
    def __init__(self):
        super(AutoEncoder, self).__init__()
        
        #encoder
        self.e1 = nn.Linear(70,28)
        self.e2 = nn.Linear(28,250)
        
        #Latent View
        self.lv = nn.Linear(250,10)
        
        #Decoder
        self.d1 = nn.Linear(10,250)
        self.d2 = nn.Linear(250,500)
        
        self.output_layer = nn.Linear(500,70)
        
    def forward(self,x):
        x = F.relu(self.e1(x))
        x = F.relu(self.e2(x))
        
        x = torch.sigmoid(self.lv(x))
        
        x = F.relu(self.d1(x))
        x = F.relu(self.d2(x))
        
        x = self.output_layer(x)
        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    

# Multi-Autoencoder, Multi-GPU
ae = AutoEncoder()
ae= nn.DataParallel(ae)
ae = ae.to(device)
print(ae)
if torch.cuda.device_count() > 0:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
epochs=64

loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(ae.parameters(), lr=1e-5)

loss = 0
epochs = 64
for epoch in range(epochs):
    
    for batch_idx, (data,target) in enumerate(trn_dataloader):
        data = torch.autograd.Variable(data)
        data=data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        pred = ae(data)
        loss = loss_func(pred,data)
        #loss = (loss.cpu().data.item())
        # Backpropagation
        loss.backward()
        optimizer.step()
        loss += loss.item()
    
    # compute the epoch training loss
    loss = loss / len(trn_dataloader)
    
    # display the epoch training loss
    print("epoch : {}/{}, recon loss = {:.8f}".format(epoch + 1, epochs, loss))





