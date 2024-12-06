import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer, Adam


## TODO: define train_epoch, test_epoch, train_model


def train_epoch(model : nn.Module, optimizer : Optimizer, train_loader : DataLoader, device : torch.device, K : int = 5) -> float:
    """ 
    Trains the model for one epoch. K is the number of samples used for the Monte Carlo estimate of the ELBO.
    """
    model.train()
    loss = 0
    for x, s in train_loader:
        x, s = x.to(device), s.to(device)
        optimizer.zero_grad()
        loss = model(x, s, K)
        loss.backward()
        optimizer.step()
    return loss.item()