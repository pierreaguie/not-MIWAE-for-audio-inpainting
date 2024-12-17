import torch
import torch.nn as nn
from src.train_test import *
from src.models import notMIWAE, LogisticMissingModel, AudioDecoder, AudioEncoder, notMIWAE, AbsoluteLogisticMissingModel
from torch.utils.data import DataLoader
from src.datasets import ClippedDataset
import argparse

device = torch.device("cuda:0")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Parser")
    
    parser.add_argument("--T", default=1024, type=int, help="Window size for dataset samples.")
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
    parser.add_argument("--latent", default=20, type=int, help="Size of the latent dimension.")
    parser.add_argument("--nepochs", type=int, default=1000, help="Number of training epochs.")
    parser.add_argument("--tensorboard", type=str, default="./tensorboard",help="Log directory for tensorboard.")
    parser.add_argument("--K", type=int,default=5, help="Parameter for the importance sampling.")
    parser.add_argument("--val", type=int,default=50, help="Interval between epoch validation.")
    parser.add_argument("--lr", type=float,default=0.001, help="Learning rate.")
    parser.add_argument("--run", type = str, default = "run", help = "Run name for tensorboard.")
    args = parser.parse_args()
    return args




if __name__ == "__main__":
    x_train = torch.load("data/big_clipped_2way/x_train.pt", weights_only=True)[:10000]
    s_train = torch.load("data/big_clipped_2way/s_train.pt", weights_only=True)[:10000]
    x_val = torch.load("data/big_clipped_2way/x_val.pt", weights_only=True)
    s_val = torch.load("data/big_clipped_2way/s_val.pt", weights_only=True)
    args = parse_arguments()

    train_dataset = ClippedDataset(x_train.to(device),s_train.to(device))
    val_dataset = ClippedDataset(x_val.to(device),s_val.to(device))

    train_loader = DataLoader(train_dataset,args.batch_size)
    val_loader = DataLoader(val_dataset,args.batch_size)

    encoder = AudioEncoder(args.T, args.latent).to(device)  
    decoder = AudioDecoder(args.T, args.latent, args.K).to(device)
    missing_model = AbsoluteLogisticMissingModel(fixed_params=True, W = 10., b = .5).to(device)
    model = notMIWAE(encoder, decoder, missing_model, args.T, args.latent, device).to(device)
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = args.lr)
    torch.autograd.set_detect_anomaly(True)

    train(model, optimizer, train_loader, val_loader, device, args.nepochs, args.K, args.val, args.tensorboard + "/" + args.run)