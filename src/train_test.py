import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Optimizer, Adam
from tqdm import tqdm
import os


def train_epoch(model : nn.Module, optimizer : Optimizer, train_loader : DataLoader, device : torch.device, epoch : int, n_epochs : int,K : int = 5,) -> float:
    """ 
    Trains the model for one epoch. K is the number of samples used for the Monte Carlo estimate of the ELBO.
    """
    model.train()
    loss = 0
    mean_loss = 0.0
    with tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}", unit="batch") as tepoch:
        for x, s in tepoch:
            x, s = x.to(device), s.to(device)
            optimizer.zero_grad()
            loss = model.loss(x, s, K)
            loss.backward()
            mean_loss += loss.item() 
            optimizer.step()
        
    return mean_loss / len(train_loader)


def val_loss_and_MSE(model, val_loader, device, K):
    model.eval()
    mean_loss = 0
    RMSE = 0
    for x, s in val_loader:
        x, s = x.to(device), s.to(device)
        loss = model.loss(x, s, K)
        mean_loss += loss.item()
        x_imputed = model.impute(x, s, K)
        MSE = torch.sum((x - x_imputed)**2, dim=1) / torch.sum(1 - s, dim=1)
        RMSE += torch.sqrt(MSE.mean()).item()
    return mean_loss / len(val_loader), RMSE / len(val_loader) 
    

def train(model : nn.Module, optimizer : Optimizer, train_loader : DataLoader, val_loader : DataLoader, device : torch.device, n_epochs : int = 500, K : int = 5, n_epochs_val : int = 20, log_dir = "./tensorboard") -> float:
    """
    Complete training function with tensorboard logging.
    """
    writer = SummaryWriter(log_dir=log_dir)
    os.makedirs("checkpoints", exist_ok=True)
    for epoch in range(n_epochs):
        train_loss = train_epoch(model, optimizer, train_loader, device, epoch, n_epochs, K)
        print(f"Training loss for epoch {epoch} is {train_loss}")
        writer.add_scalar("Loss/Train", train_loss, epoch)
        if epoch % n_epochs_val == 0:
            val_loss, val_RMSE = val_loss_and_MSE(model, val_loader, device, K)
            print(f"Validation loss / Root Mean Squared Error for epoch {epoch} is {val_loss}/{val_RMSE}")
            # Save the checkpoints
            torch.save(model.state_dict(), f'checkpoints/model_{epoch}.pth')
            writer.add_scalar("Loss/Validation", val_loss, epoch)
            writer.add_scalar("RMSE/Validation", val_RMSE, epoch)
    
    writer.close()
    
            
            
            
    
    
    