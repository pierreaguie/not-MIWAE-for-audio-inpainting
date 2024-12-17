import torch
import torch.nn as nn
from src.train_test import *
from src.models import notMIWAE, notMIWAE, AudioEncoder, AudioDecoder, AbsoluteLogisticMissingModel
import argparse
from torchaudio.transforms import Resample
import matplotlib.pyplot as plt
import torchaudio
import soundfile as sf
import librosa
import numpy as np
from src.utils import soft_clipping


device = torch.device("cuda:0")
torchaudio.set_audio_backend("sox")
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

def cut_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path,frame_offset=100000,num_frames=890000)
    resampler = Resample(orig_freq=sample_rate, new_freq=16000)
    waveform = resampler(waveform)[:,:312*1024] #We choose this so it can cut the audio exactly in 1024 size frames
    sample_rate = 16000
    frames = []
    original_norms = []
    for i in range(312):
        x = waveform[:,i*1024:(i+1)*1024]
        
        
        original_norm = torch.norm(x,p=float("inf"),keepdim=True).to(device)
        x = torch.nn.functional.normalize(x,p=float("inf"),dim=1)
        original_norms.append(original_norm)
        s = soft_clipping(x,10.0,0.6)
        frames.append((x.to(device),s.to(device)))
    return frames, original_norms, waveform
        
        
    
    
    

if __name__ == "__main__":
    
    
    
    args = parse_arguments()
    frames, original_norms, original_audio = cut_audio("assets/audio/1819.wav")
    
    
    
    encoder = AudioEncoder(args.T, args.latent).to(device)  
    decoder = AudioDecoder(args.T, args.latent, args.K).to(device)
    missing_model = AbsoluteLogisticMissingModel(fixed_params=True, W = 10., b = 0.6).to(device)
    model = notMIWAE(encoder, decoder, missing_model, args.T, args.latent, device).to(device)    
    state_dict = torch.load("checkpoints/model_995.pth")
    model.load_state_dict(state_dict)
    
    res_frames = []
    for i in range(len(frames)):
        x, s = frames[i]
        x_imputed = model.impute(x, s, 1)
        original_norm = original_norms[i]
        res_frames.append(x_imputed * original_norm)
    reconstructed_audio = torch.cat(res_frames, dim=1)
    
    
    
    
    #torchaudio.save("original_audio.wav", original_audio,sample_rate=16000,format="wav")
    sf.write("original_audio.wav",original_audio.t().cpu().numpy(),16000)
    sf.write("reconstructed_audio.wav", reconstructed_audio.t().cpu().detach().numpy(),16000)
    
    y_original, sr_original = librosa.load("original_audio.wav", sr=None)  # Load with original sample rate
    y_reconstructed, sr_reconstructed = librosa.load("reconstructed_audio.wav", sr=None)

    # Compute spectrograms (e.g., Short-Time Fourier Transform)
    D_original = librosa.amplitude_to_db(np.abs(librosa.stft(y_original)), ref=np.max)
    D_reconstructed = librosa.amplitude_to_db(np.abs(librosa.stft(y_reconstructed)), ref=np.max)
    
    plt.figure(figsize=(12, 6))

    # Original Audio Spectrogram
    plt.subplot(1, 2, 1)
    librosa.display.specshow(D_original, sr=sr_original, x_axis='time', y_axis='log')
    plt.title('Original Audio Spectrogram')
    plt.colorbar(format='%+2.0f dB')

    # Reconstructed Audio Spectrogram
    plt.subplot(1, 2, 2)
    librosa.display.specshow(D_reconstructed, sr=sr_reconstructed, x_axis='time', y_axis='log')
    plt.title('Reconstructed Audio Spectrogram')
    plt.colorbar(format='%+2.0f dB')

    plt.tight_layout()
    plt.show()
    plt.savefig("Spectrograms.pdf",format="pdf")
    spectral_convergence = np.linalg.norm(D_original - D_reconstructed) / np.linalg.norm(D_original)
    print(f"Spectral Convergence between the original and reconstructed audio: {spectral_convergence:.6f}")
    
