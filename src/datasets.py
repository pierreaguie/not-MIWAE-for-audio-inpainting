import torch
from torch.utils.data import Dataset
import numpy as np
import os
import torchaudio
from torchaudio.transforms import Resample
import random
from src.utils import soft_clipping, hard_clipping, normalize


class ClippedDataset(Dataset):
    """ 
    ClippedDataset class. A dataset of clipped signals.

    Args:
    --------
    x (torch.Tensor): the input, true signals
    s (torch.Tensor): the clipping masks (s = 1 if the signal is clipped, s = 0 otherwise)

    When passed in notMIWAE, the input to the autoencoder is x * s, so that the missing values are not used to train the model.
    """

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



def generate_synthetic_dataset(N, T, K, clipping_model : str = "soft", W : float = 50, thresh : float = .8, sr : float = 16000, max_freq : int = 2000):
    """ 
    Generates a synthetic dataset of N samples of length T. Each sample is a mixture of K sinusoids, clipped using the chosen clipping model.
    """
    x = torch.zeros(N, T)
    s = torch.zeros(N, T)
    for i in range(N):
        for k in range(K):
            A = torch.randn(1)
            f = np.random.randint(1, max_freq)
            phi = torch.rand(1) * 2 * np.pi
            x[i] += A * torch.sin(2 * np.pi * f * torch.arange(T) / sr + phi)
        x[i] = normalize(x[i])
        if clipping_model == "soft":
            s[i] = soft_clipping(x[i], W, thresh)
        elif clipping_model == "hard":
            s[i] = hard_clipping(x[i], thresh)

    return ClippedDataset(x, s)    


def load_dataset(dataset_dir : str, n_samples : int, window_size : int, target_sample_rate : int = 16000, clipping_model : str = "soft", W : float = 50, thresh : float = .8):
    x = torch.zeros(n_samples, window_size)
    s = torch.zeros(n_samples, window_size)
    list_files = os.listdir(dataset_dir)
    selected_files = random.choices(list_files, k=n_samples//10)
    
    for i,file in zip(range(n_samples),selected_files):
        
        file_path = os.path.join(dataset_dir, file)
        
        start_frame = random.randint(660000,1320000)
        # Take ten seconds of audio
        waveform, sample_rate = torchaudio.load(file_path,frame_offset=start_frame,num_frames=440000)
        
        # Resample and normalize 
        if sample_rate != target_sample_rate:
            resampler = Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
            waveform = resampler(waveform)
            sample_rate = target_sample_rate
        
        # Take 10 parts of the audio
        num_samples = waveform.size(1)
        start_points = random.sample(range(0, num_samples - window_size),10)
        
        
        for j in range(10):
            x[i*10+j,:] = torch.nn.functional.normalize(waveform[:,start_points[j]:start_points[j]+window_size], p=float("inf"),dim=1)
        
        
            if clipping_model == "soft":
                s[i*10+j] = soft_clipping(x[i*10+j], W, thresh)
            elif clipping_model == "hard":
                s[i*10+j] = hard_clipping(x[i*10+j], thresh)
        
    return ClippedDataset(x, s)
