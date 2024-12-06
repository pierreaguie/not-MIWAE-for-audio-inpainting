import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Bernoulli, Independent

import numpy as np


class notMIWAE(nn.Module):

    def __init__(self, 
                 encoder : nn.Module,
                 decoder : nn.Module, 
                 missing_model : nn.Module,
                 encoder_input_dim : int,
                 encoder_output_dim : int, 
                 decoder_output_dim : int, 
                 latent_dim : int):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.missing_model = missing_model

        self.encoder_input_dim = encoder_input_dim
        self.encoder_output_dim = encoder_output_dim
        self.decoder_output_dim = decoder_output_dim
        self.latent_dim = latent_dim

        self.q_mu = nn.Linear(encoder_output_dim, latent_dim)
        self.q_logvar = nn.Linear(encoder_output_dim, latent_dim)
        self.p_mu = nn.Linear(decoder_output_dim, encoder_input_dim)
        self.p_logvar = nn.Linear(decoder_output_dim, encoder_input_dim)

        # Prior distribution
        self.p_z = Independent(Normal(torch.zeros(1), torch.ones(1)), 1)


    def forward(self, x : torch.Tensor, s : torch.Tensor, K : int = 1) -> torch.Tensor:
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

        log_p_x_given_z, log_q_z_given_x, log_p_s_given_x, log_p_z, _ = self.log_probabilities_and_missing_data(x, s, K)
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
        log_p_x_given_z, log_q_z_given_x, log_p_s_given_x, log_p_z, x_missing = self.log_probabilities_and_missing_data(x, s, K)
        imp_weights = F.softmax(log_p_x_given_z + log_p_s_given_x - log_q_z_given_x + log_p_z, dim = 0)

        return torch.einsum('ki,kij->ij', imp_weights, x_missing) + x_observed
    
    
    def log_probabilities_and_missing_data(self, x : torch.Tensor, s : torch.Tensor, K : int = 1):
        """         
        Inputs:
        ------
        - x: Tensor of shape (batch_size, input_dim)
        - s: Tensor of shape (batch_size, input_dim) with 1s where x is observed and 0s where x is missing
        - K: Number of samples to draw from the posterior (default = 1)

        Outputs:
        -------
        - log_p_x_given_z: Tensor of shape (K, batch_size) with the log probabilities of the observed data
        - log_q_z_given_x: Tensor of shape (K, batch_size) with the log probabilities of the posterior
        - log_p_s_given_x: Tensor of shape (K, batch_size) with the log probabilties of the missing model
        - log_p_z: Tensor of shape (K, batch_size) with the log probabilities of the prior
        - x_missing: Tensor of shape (K, batch_size, input_dim) with the missing data and 0s where x is observed        
        """
        
        # s[i,j] = 1 iff x[i,j] is observed. If x[i,j] is missing, pad with 0.
        x_observed = s * x                                                # Size (bs,input_dim)

        # Encoder: q(z|x_observed)
        h = self.encoder(x_observed)
        mu_z = self.q_mu(h)
        logvar_z = self.q_logvar(h)

        # Sampling z from q(z|x_observed) K times
        q_z_given_x = Independent(Normal(loc = mu_z, scale = torch.exp(0.5 * logvar_z)), 1)
        z = q_z_given_x.rsample([K])                                      # Size (K,bs,latent_dim)
        
        # Decoder: p(x|z_k) for all k = 1, ..., K
        h = self.decoder(z)
        mu_x = self.p_mu(h)
        logvar_x = self.p_logvar(h)
        p_x_given_z = Independent(Normal(mu_x, torch.exp(0.5 * logvar_x)),1)

        # Sample missing data
        x_missing = p_x_given_z.rsample() * (1 - s)
        x_imputed = x_observed + x_missing

        # Missing model: p(s|x_observed, x_missing)
        p_s_given_x = Independent(Bernoulli(self.missing_model(x_imputed)),1)

        # Log probabilities:
        x_observed_k = torch.Tensor.repeat(x_observed, [K,1,1])            # Size (K,bs,input_dim)
        s_k = torch.Tensor.repeat(s, [K,1,1])                              
        
        log_p_x_given_z = p_x_given_z.log_prob(x_observed_k)               # Size (K, bs)
        log_q_z_given_x = q_z_given_x.log_prob(z)
        log_p_s_given_x = p_s_given_x.log_prob(s_k)
        log_p_z = self.p_z.log_prob(z)
        
        return log_p_x_given_z, log_q_z_given_x, log_p_s_given_x, log_p_z, x_missing



class LogisticMissingModel(nn.Module):

    def __init__(self, fixed_params : bool = False, W : float = 50., b : float = 0.8):
        """ 
        Missing model of the form p(s_j|x_j^m) = sigmoid(W * (abs(x_j^m) - b))
        """
        super().__init__()

        self.W = nn.Parameter(torch.tensor(W), requires_grad = not fixed_params)
        self.b = nn.Parameter(torch.tensor(b), requires_grad = not fixed_params)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.W * (torch.abs(x) - self.b))
    


## TODO: define the encoder, decoder