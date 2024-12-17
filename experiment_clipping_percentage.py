import torch
import torch.nn as nn
from src.train_test import *
from src.models import notMIWAE, LogisticMissingModel, AudioDecoder, AudioEncoder, notMIWAE, AudioDecoderV2, AudioEncoderV2, AbsoluteLogisticMissingModel
from torch.utils.data import DataLoader
from src.datasets import ClippedDataset
from src.utils import normalize
import argparse
import matplotlib.pyplot as plt
from src.utils import soft_clipping
device = torch.device("cuda:0")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Parser")
    
    parser.add_argument("--T", default=1024, type=int, help="Window size for dataset samples.")
    parser.add_argument("--batch_size", default=256, type=int, help="Batch size.")
    parser.add_argument("--latent", default=100, type=int, help="Size of the latent dimension.")
    parser.add_argument("--nepochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--tensorboard", type=str, default="./tensorboard",help="Log directory for tensorboard.")
    parser.add_argument("--K", type=int,default=1, help="Parameter for the importance sampling.")
    parser.add_argument("--val", type=int,default=5, help="Interval between epoch validation.")
    parser.add_argument("--lr", type=float,default=1e-4, help="Learning rate.")
    parser.add_argument("--run", type = str, default = "run", help = "Run name for tensorboard.")
    args = parser.parse_args()
    return args




if __name__ == "__main__":
    
    clipping_percentage = [0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    """
    
    
    x_train = torch.load("data/musicnet_renorm_reclip/x_train.pt")
    x_val = torch.load("data/musicnet_renorm_reclip/x_val.pt")
    
    s_train = soft_clipping(x_train,10.0,0.5)
    s_val = soft_clipping(x_val,10.0,.5)
    
    torch.save(s_train,"data/musicnet_renorm_reclip/s_train.pt")
    torch.save(s_train,"data/musicnet_renorm_reclip/s_val.pt")
    print("saved")
    """
    
    
    x_train = torch.load("data/musicnet_renorm_reclip/x_train.pt")
    
    x_val = torch.load("data/musicnet_renorm_reclip/x_val.pt")
    
    args = parse_arguments()

    
    list_best_RMSE = []
    list_missing_percentage = []
    encoder = AudioEncoderV2(args.T, args.latent).to(device)  
    decoder = AudioDecoderV2(args.T, args.latent, args.K).to(device)
    for clipping in clipping_percentage:
        s_train = soft_clipping(x_train,10.0,clipping)
        s_val = soft_clipping(x_val,10.0,clipping)
        
        num_zeros_val = (s_val == 0).sum().item()
        
        percentage_val = (num_zeros_val / s_val.numel()) * 100
        list_missing_percentage.append(percentage_val)
        
        train_dataset = ClippedDataset(x_train.to(device),s_train.to(device))
        val_dataset = ClippedDataset(x_val.to(device),s_val.to(device))

        train_loader = DataLoader(train_dataset,args.batch_size)
        val_loader = DataLoader(val_dataset,args.batch_size)
        
        
        missing_model = AbsoluteLogisticMissingModel(fixed_params=True, W = 10., b = clipping).to(device)
        model = notMIWAE(encoder, decoder, missing_model, args.T, args.latent, device).to(device)
    
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = args.lr)
        torch.autograd.set_detect_anomaly(True)

        best_RMSE = train_and_give_best_RMSE(model, optimizer, train_loader, val_loader, device, args.latent, args.nepochs, args.K, args.val, args.tensorboard + "/" + args.run)
        list_best_RMSE.append(best_RMSE)
    plt.figure(figsize=(8, 6))
    plt.plot(clipping_percentage, list_best_RMSE, marker='o', linestyle='-', color='b', label='Best RMSE')
    legend_entries = [f"Clipping of {clipping_percentage[i]} : {list_missing_percentage[i]:.2f}% of missing values" for i in range(len(clipping_percentage))]
    legend_text = "\n".join(legend_entries)
    
    plt.xlabel('Clipping', fontsize=12)
    plt.ylabel('Best RMSE', fontsize=12)
    plt.title('Best RMSE vs Clipping', fontsize=14)
    plt.legend([legend_text],loc="upper left",fontsize="small")
    plt.grid(True)
    output_file = 'best_RMSE_vs_clipping.png'
    plt.savefig(output_file, dpi=300)
    plt.show()
    print(f"Plot saved as {output_file}")
    
        
    