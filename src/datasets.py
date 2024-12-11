import torch
from torch.utils.data import Dataset, DataLoader
from torchaudio.datasets import MUSDB_HQ
import numpy as np
import os
import torchaudio
from torchaudio.transforms import Resample
import random
from src.utils import soft_clipping, hard_clipping, normalize


class ClippedDataset(Dataset):

    def __init__(self, x : torch.Tensor, s : torch.Tensor):
        self.x = x
        self.s = s
        self.N, self.T = x.shape

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.x[idx], self.s[idx]
    

def train_val_test_split(dataset, train_size : float = .8, val_size : float = .1, test_size : float = .1):
    N = len(dataset)
    train_size = int(N * train_size)
    val_size = int(N * val_size)
    test_size = N - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    return train_dataset, val_dataset, test_dataset


def collate(batch):
    x, s = zip(*batch)
    x = torch.stack(x)
    s = torch.stack(s)
    return x, s



def generate_synthetic_dataset(N, T, K, clipping_model : str = "soft", W : float = 50, thresh : float = .8):
    """ 
    Generates a synthetic dataset of N samples of length T. Each sample is a mixture of K sinusoids, clipped using the chosen clipping model.
    """
    x = torch.zeros(N, T)
    s = torch.zeros(N, T)
    for i in range(N):
        for k in range(K):
            A = torch.rand(1) * 2 - 1
            f = torch.randint(1, T // 2, (1,))
            phi = torch.rand(1) * 2 * np.pi
            x[i] += A * torch.sin(2 * np.pi * f * torch.arange(T).float() / T + phi)
        x[i] = normalize(x[i])
        if clipping_model == "soft":
            s[i] = soft_clipping(x[i], W, thresh)
        elif clipping_model == "hard":
            s[i] = hard_clipping(x[i], thresh)
        x[i] = (1-s[i]) * x[i]

    return ClippedDataset(x, s)    


def load_dataset(dataset_dir : str, n_samples : int, window_size : int, target_sample_rate : int = 16000, clipping_model : str = "soft", W : float = 50, thresh : float = .8):
    x = torch.zeros(n_samples, window_size)
    s = torch.zeros(n_samples, window_size)
    list_files = os.listdir(dataset_dir)
    selected_files = random.choices(list_files, k=n_samples)
    
    for i,file in zip(range(n_samples),selected_files):
        file_path = os.path.join(dataset_dir, file)
        waveform, sample_rate = torchaudio.load(file_path)
        
        if sample_rate != target_sample_rate:
            resampler = Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
            waveform = resampler(waveform)
            sample_rate = target_sample_rate
            

        num_samples = waveform.size(1)
        start_point = random.randint(0, num_samples - window_size)
        segment = waveform[:,start_point:start_point+window_size]
        x[i,:] = segment
        x[i] = normalize(x[i])
        if clipping_model == "soft":
            s[i] = soft_clipping(x[i], W, thresh)
        elif clipping_model == "hard":
            s[i] = hard_clipping(x[i], thresh)
        
    return ClippedDataset(x, s)
        
            
        
        
    
    










## TODO: add a function to load a real audio dataset (e.g. MUSDB)