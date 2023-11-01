
DIM_FROM    = 3

DIM_TO      = 2

REM_DIM     = 1

TOTAL_VECS  = 10000

SPLITS      = [ 0.8, 0.2, 0.2 ]

SEED        = 42


#
# imports
#
import os
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
#from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import pandas as pd

#
# create a bunch of candidate neural net architectures
#

class Linear1(nn.Module):
    '''Define a 1-layer neural network, no drop out, no activation'''
    def __init__(self):
        super(Linear1, self).__init__()
        self.fc = nn.Linear(DIM_FROM, DIM_TO)
    def forward(self, x):
        output = self.fc(x)
        return output

def train(model, device, train_loader, optimizer, epoch):
    '''Define the train function'''   
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()         
        output = model(data)          
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
    
def validate(model, device, validate_loader): 
    '''Define the validation function'''
    model.eval()
    validate_loss = 0
    with torch.no_grad():
        for data, target in validate_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            validate_loss += F.mse_loss(output, target, reduction='sum').item()  # sum up batch loss

    validate_loss /= len(validate_loader.dataset)
    return validate_loss

def test(model, device, test_loader):
    '''Define the test function'''
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.mse_loss(output, target, reduction='sum').item()  # sum up batch loss

    test_loss /= len(test_loader.dataset)
    return test_loss


def main():
    
    print("torch version=", torch.version.__version__)

    #
    # set ran seeds for reproducibility
    #
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # choose accelerator if any
    device = torch.device("cpu")

    #
    # initiaize dataset in numpy
    #
    dset_in = np.random.rand( TOTAL_VECS, DIM_FROM )
    dset_out = np.array( [ np.array( list(el[0:REM_DIM]) + list(el[REM_DIM+1:]) ) for el in dset_in ] )
    # sanity check
    print("dataset sanity check...")
    print(dset_in.shape, dset_in.dtype, dset_in[0].dtype)
    print(dset_out.shape, dset_out.dtype, dset_out[0].dtype)
    print(dset_in[0], dset_out[0])
    print(dset_in[-1], dset_out[-1])
    print()

    # use sci-kit to create train/validate/test splits
    x_train, x_test, y_train, y_test = train_test_split(dset_in, dset_out, test_size=0.4)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5)
    # sanity check
    print("tr/val/test split sanity check:", x_train.shape, x_val.shape, x_test.shape)
    print()

    # create the model with parameters
    model = Linear1().to(device)

    #
    # convert numpy dataset to torch tensors
    #
    x_train_tensors = torch.Tensor(x_train)
    y_train_tensors = torch.Tensor(y_train)
    train_tensors = TensorDataset(x_train_tensors,y_train_tensors)

    x_validate_tensors = torch.Tensor(x_val)
    y_validate_tensors = torch.Tensor(y_val)
    validate_tensors = TensorDataset(x_validate_tensors,y_validate_tensors)

    x_test_tensors = torch.Tensor(x_test)
    y_test_tensors = torch.Tensor(y_test)
    test_tensors = TensorDataset(x_test_tensors,y_test_tensors)

    # create the batch loaders
    train_loader = DataLoader(train_tensors,batch_size=10 )
    validate_loader = torch.utils.data.DataLoader(validate_tensors, batch_size=10 )
    test_loader = torch.utils.data.DataLoader(test_tensors, batch_size=10 )
   
    # prepare training 
    optimizer = optim.Adadelta(model.parameters(), lr=1.0)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

    # the training loop for this model
    for epoch in range(1, 10):
        tr_loss = train(model, device, train_loader, optimizer, epoch)
        val_loss = validate(model, device, validate_loader)
        print("Training losses=", tr_loss, val_loss)
        scheduler.step()

    # test the model
    avg_test_loss = test(model, device, test_loader)
    print("avg test loss=", avg_test_loss)


if __name__=="__main__":
    main()
