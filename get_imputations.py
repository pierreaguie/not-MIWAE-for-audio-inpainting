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
    
    nbr_imputations = 4
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
    s_train = soft_clipping(x_train,10.0,0.6)
    x_val = torch.load("data/musicnet_renorm_reclip/x_val.pt")
    s_val = soft_clipping(x_val,10.0,0.6)
    args = parse_arguments()

    
    
    
    encoder = AudioEncoderV2(args.T, args.latent).to(device)  
    decoder = AudioDecoderV2(args.T, args.latent, args.K).to(device)
    missing_model = AbsoluteLogisticMissingModel(fixed_params=True, W = 10., b = 0.6).to(device)
    model = notMIWAE(encoder, decoder, missing_model, args.T, args.latent, device).to(device)    
    train_dataset = ClippedDataset(x_train.to(device),s_train.to(device))
    val_dataset = ClippedDataset(x_val.to(device),s_val.to(device))
    train_loader = DataLoader(train_dataset,args.batch_size)
    val_loader = DataLoader(val_dataset,args.batch_size)
        
    
        
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = args.lr)
    torch.autograd.set_detect_anomaly(True)
    state_dict = torch.load("checkpoints/model_995.pth")
    model.load_state_dict(state_dict)
    #train(model, optimizer, train_loader, val_loader, device, args.nepochs, args.K, args.val, args.tensorboard + "/" + args.run)
    original_data = []
    original_s = []
    imputation_results = []
    for imputation in range(nbr_imputations):
        idx = torch.randint(0,len(x_train),(1,)).to(device)
        
        x= x_train[idx].view(1,1024).to(device)
        print(x.shape)
        original_data.append(x[0,400:600])
        s = s_train[idx].view(1,1024).to(device)
        print(s.shape)
        original_s.append(s[0,400:600])
        x_imputed = model.impute(x,s,1)
        imputation_results.append(x_imputed[0,400:600])
        
        
    
    fig, axs = plt.subplots(2, nbr_imputations//2, figsize=(10, 8))  # Create 2x2 subplots
    fig.suptitle("Imputed vs Original Data", fontsize=16)
    axs =axs.flatten()
    
    for i in range(nbr_imputations):
        ax = axs[i]
        s = original_s[i].view(200).cpu().numpy()
        imputed = imputation_results[i].t().cpu().detach().numpy()
        ax.plot(original_data[i].t().cpu().numpy(), label="Original data")
        imputed_indices = [j for j, v in enumerate(s) if v == 0]
        imputed_values = [imputed[j] for j, v in enumerate(s) if v == 0]
        ax.scatter(imputed_indices, imputed_values, label="Imputed data (s=0)", color = "r",marker="o")
        ax.set_title(f"Plot {i+1}")
        ax.legend(loc="lower right")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_file = 'imputation_vs_original.pdf'
    plt.savefig(output_file, format="pdf")
    plt.show()
    print(f"Plot saved as {output_file}")
    
        
    