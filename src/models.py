import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Bernoulli, Independent

import numpy as np


class AudioEncoder(nn.Module):

    def __init__(self, T : int, latent_dim : int, dropout : float = .3):
        super().__init__()

        self.T = T
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, stride=2, padding=7), 
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=15, stride=2, padding=7), 
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=15, stride=2, padding=7),  
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=15, stride=2, padding=7), 
            nn.ReLU()
        )

        self.flatten = nn.Flatten()        
        self.fc_mu = nn.Linear((self.T//16) * 128, self.latent_dim)
        self.fc_logvar = nn.Linear((self.T//16) * 128, self.latent_dim)


    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.encoder(x)
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    

class AudioDecoder(nn.Module):

    def __init__(self, T : int, latent_dim : int, K : int):
        super().__init__()

        self.T = T
        self.latent_dim = latent_dim

        self.fc = nn.Linear(latent_dim, (self.T//16) * 128)

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=15, stride=2, padding=7, output_padding=1),  # (64 -> 128)
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=15, stride=2, padding=7, output_padding=1),  # (128 -> 256)
            nn.ReLU()
        )
        self.decoder_mu = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=15, stride=2, padding=7, output_padding=1),  # (256 -> 512)
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=15, stride=2, padding=7, output_padding=1)
        )
        self.decoder_logvar = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=15, stride=2, padding=7, output_padding=1),  # (256 -> 512)
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=15, stride=2, padding=7, output_padding=1)
        )

    def forward(self, z : torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = x.view(x.size(0), 128, self.T//16)
        x = self.decoder(x)
        mu = self.decoder_mu(x).squeeze(1)
        logvar = self.decoder_logvar(x).squeeze(1)
        return mu, logvar


class notMIWAE(nn.Module):

    def __init__(self, 
                 encoder : nn.Module,
                 decoder : nn.Module, 
                 missing_model : nn.Module,
                 T : int,
                 latent_dim : int,
                 device : torch.device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.missing_model = missing_model

        self.T = T
        self.latent_dim = latent_dim
        self.device = device

        # Prior distribution
        self.p_z = Independent(Normal(torch.zeros(self.latent_dim).to(device), torch.ones(self.latent_dim).to(device)), 1)

        
    def forward(self, x : torch.Tensor, s : torch.Tensor) -> torch.Tensor:
        """ 
        Returns the mean and the log variance of the posterior distribution of z given x
         
        Inputs:
        ------
        - x: Tensor of shape (batch_size, T)
        - s: Tensor of shape (batch_size, T) with 1s where x is observed and 0s where x is missing

        Outputs:
        -------
        - mu_z: Tensor of shape (batch_size, latent_dim) with the mean of the posterior distribution of z given x
        - logvar_z: Tensor of shape (batch_size, latent_dim) with the log variance of the posterior distribution of z given x
        """
        # s[i,j] = 1 iff x[i,j] is observed. If x[i,j] is missing, pad with 0.
        x_observed = s * x                                                # Size (bs, T)

        # Encoder: q(z|x_observed)
        mu_z, logvar_z = self.encoder(x_observed)
        return mu_z, logvar_z

    def loss(self, x : torch.Tensor, s : torch.Tensor, K : int = 1) -> torch.Tensor:
        """
        Computes the not-MIWAE loss
        
        Inputs:
        ------
        - x: Tensor of shape (batch_size, input_dim)
        - s: Tensor of shape (batch_size, input_dim) with 1s where x is observed and 0s where x is missing
        - K: Number of samples to draw from the posterior (default = 1)

        Outputs:
        -------
        - loss: Tensor of shape (1) with the not-MIWAE loss
        """

        log_p_x_given_z, log_q_z_given_x, log_p_s_given_x, log_p_z, _, _ = self.log_probabilities_and_missing_data(x, s, K)
        loss = -torch.mean(torch.logsumexp(log_p_x_given_z + log_p_s_given_x - log_q_z_given_x + log_p_z - np.log(K), dim = 0))
        return loss

    
    def impute(self, x : torch.Tensor, s : torch.Tensor, K : int = 1) -> torch.Tensor:
        """ 
        Imputes missing values in x using the trained model
        
        Inputs:
        ------
        - x: Tensor of shape (batch_size, input_dim)
        - s: Tensor of shape (batch_size, input_dim) with 1s where x is observed and 0s where x is missing
        - K: Number of samples to draw from the posterior (default = 1)

        Outputs:
        -------
        - x_imputed: Tensor of shape (batch_size, input_dim) with the imputed missing values and observed values
        """

        x_observed = s * x
        log_p_x_given_z, log_q_z_given_x, log_p_s_given_x, log_p_z, _, mu_x = self.log_probabilities_and_missing_data(x, s, K)
        imp_weights = F.softmax(log_p_x_given_z + log_p_s_given_x - log_q_z_given_x + log_p_z, dim = 0)

        return torch.einsum('ki,kij->ij', imp_weights, mu_x*(1-s)) + x_observed
    
    
    def log_probabilities_and_missing_data(self, x : torch.Tensor, s : torch.Tensor, K : int = 1):
        """         
        Inputs:
        ------
        - x: Tensor of shape (batch_size, T)
        - s: Tensor of shape (batch_size, T) with 1s where x is observed and 0s where x is missing
        - K: Number of samples to draw from the posterior (default = 1)

        Outputs:
        -------
        - log_p_x_given_z: Tensor of shape (K, batch_size) with the log probabilities of the observed data
        - log_q_z_given_x: Tensor of shape (K, batch_size) with the log probabilities of the posterior
        - log_p_s_given_x: Tensor of shape (K, batch_size) with the log probabilties of the missing model
        - log_p_z: Tensor of shape (K, batch_size) with the log probabilities of the prior
        - x_missing: Tensor of shape (K, batch_size, T) with the missing data and 0s where x is observed    
        - mu_x: Tensor of shape (K, batch_size, T) with the mean of the decoder distribution    
        """
        
        # s[i,j] = 1 iff x[i,j] is observed. If x[i,j] is missing, pad with 0.
        x_observed = s * x                                                # Size (bs, T)

        # Encoder: q(z|x_observed)
        mu_z, logvar_z_unclamped = self.encoder(x_observed)
        logvar_z = torch.clamp(logvar_z_unclamped, max = 5, min = -5)
        
        # Sampling z from q(z|x_observed) K times
        q_z_given_x = Independent(Normal(loc = mu_z, scale = torch.exp(0.5 * logvar_z)), 1)
        z = q_z_given_x.rsample([K]).to(self.device)                                      # Size (K, bs, latent_dim)
        
        # Decoder: p(x|z_k) for all k = 1, ..., K
        mu_x = torch.zeros((K,x.shape[0],self.T),device=self.device)
        logvar_x = torch.zeros((K,x.shape[0],self.T),device=self.device)
        for i in range(K):
            mu_x[i], logvar_x_unclamped = self.decoder(z[i])
            logvar_x[i] = torch.clamp(logvar_x_unclamped, max = 5, min = -5)
        p_x_given_z = Independent(Normal(mu_x, torch.exp(0.5 * logvar_x)),1)

        # Sample missing data
        x_missing = p_x_given_z.rsample() * (1 - s)
        x_imputed = x_observed + x_missing
        
        # Missing model: p(s|x_observed, x_missing)
        p_s_given_x = Independent(Bernoulli(logits = self.missing_model(x_imputed)),1)

        # Log probabilities:
        x_observed_k = torch.Tensor.repeat(x_observed, [K,1,1])            # Size (K, bs, T)
        s_k = torch.Tensor.repeat(s, [K,1,1])                              
        log_p_x_given_z = p_x_given_z.log_prob(x_observed_k)               # Size (K, bs)
        log_q_z_given_x = q_z_given_x.log_prob(z)
        log_p_s_given_x = p_s_given_x.log_prob(s_k)
        log_p_z = self.p_z.log_prob(z)

        return log_p_x_given_z, log_q_z_given_x, log_p_s_given_x, log_p_z, x_missing, mu_x



class LogisticMissingModel(nn.Module):

    def __init__(self, fixed_params : bool = False, W : float = 1., b : float = 0.5):
        """ 
        Missing model of the form p(s_j = 1|x_j^m) = sigmoid(-W * (x_j^m - b))      (i.e. the probability of x_j observed is low if x_j > b) 
        """
        super().__init__()

        self.W = nn.Parameter(torch.tensor(W), requires_grad = not fixed_params)
        self.b = nn.Parameter(torch.tensor(b), requires_grad = not fixed_params)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """ 
        Returns the logits of the Bernoulli distribution
        """
        return torch.clamp(- self.W * (x - self.b), min = -5, max = 5)
    


class AbsoluteLogisticMissingModel(nn.Module):

    def __init__(self, fixed_params : bool = False, W : float = 50., b : float = 0.8):
        """ 
        Missing model of the form p(s_j = 1|x_j^m) = sigmoid(-W * (|x_j^m| - b))      (i.e. the probability of x_j observed is low if |x_j| > b) 
        """
        super().__init__()

        self.W = nn.Parameter(torch.tensor(W), requires_grad = not fixed_params)
        self.b = nn.Parameter(torch.tensor(b), requires_grad = not fixed_params)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """ 
        Returns the logits of the Bernoulli distribution
        """
        return - self.W * (torch.abs(x) - self.b)