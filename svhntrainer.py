import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import SVHN
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from ResnetModel import resnet34
from quasiadam import quasiAdam
from pdb import set_trace
import pickle as pkl
batch_size_train = 128
batch_size_test = 1024


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def accuracy(outputs, labels):
    pred = outputs.data.max(1, keepdim=True)[1]
    return pred.eq(labels.data.view_as(pred)).sum()/outputs.shape[0]*100

def evaluate(model, val_loader):
    val_acc = 0.
    val_loss = 0.
    for batch in val_loader:
        lossDict = model.validation_step(batch)
        val_loss+= lossDict['val_loss']
        val_acc += lossDict['val_acc']
    return {'val_loss':val_loss , 'val_acc':val_acc/100}


trainLoader, testLoader = torch.utils.data.DataLoader(
                                                            torchvision.datasets.SVHN(
                                                                  root='../data/', split="train", download=True, transform=torchvision.transforms.ToTensor()),
                                                                  batch_size=batch_size_train, shuffle=True),torch.utils.data.DataLoader(torchvision.datasets.SVHN(
                                                                  root='../data/', download=True, split="test", transform=torchvision.transforms.ToTensor()),
                                                                  batch_size=batch_size_train, shuffle=True)




class SvhnModel(nn.Module):
    """Feedfoward neural network with 6 hidden layer"""
    def __init__(self):
        super().__init__()
        # hidden layer
        self.model = resnet34()
    def forward(self, xb):
        # Flatten the image tensors
        out = self.model(xb)
        return out
    
    def training_step(self, batch):
        images, labels = batch 
        out = self.model(images.to(device))    # Generate predictions
        loss = F.cross_entropy(out, labels.to(device)) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        with torch.no_grad():
            images, labels = batch 
            out = self.model(images.to(device))                    # Generate predictions
            loss = F.cross_entropy(out, labels.to(device))   # Calculate loss
            acc = accuracy(out, labels.to(device))           # Calculate accuracy
        return {'val_loss': loss, 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        epoch_loss = torch.stack([x['val_loss'] for x in outputs]).mean()   # Combine losses
        epoch_acc = torch.stack([x['val_acc'] for x in outputs]).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))


history = []
epochs = 50
lr = 1e-3

opts = ['adam', 'quasiAdam']
for opt in opts:
    model = SvhnModel().to(device)
    if opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr)
    elif opt == 'quasiAdam':
        optimizer = quasiAdam(model.parameters(), lr, foreach=False)
        
    testlosses = []
    try:
        for epoch in range(epochs):
            # Training Phase 
            for batch in trainLoader:
                loss = model.training_step(batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                result = evaluate(model, testLoader)
                testlosses.append(result['val_acc'])
                model.epoch_end(epoch, result)
        
        
        with open('./results/'+opt+'.pkl', 'wb') as handle:
            pkl.dump(testlosses, handle, protocol=pkl.HIGHEST_PROTOCOL)
    except KeyboardInterrupt:
        with open('./results/SVHNResults/'+opt+'.pkl', 'wb') as handle:
            print('Keyboard interrupted .. Saving to'+'./results/SVHNResults/'+opt+'.pkl  and exiting')
            pkl.dump(testlosses, handle, protocol=pkl.HIGHEST_PROTOCOL)