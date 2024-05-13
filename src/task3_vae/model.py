import torch
import torch.nn as nn
import numpy as np
import numpy.typing as npt

class VAE(nn.Module):

    def __init__(self, d_in:int, d_latent:int, d_hidden_layer:int, device):
        """Initialiize the VAE.

        Args:
            d_in (int): Input dimension
            d_latent (int): Latent dimension.
            d_hidden_layer (int): Number of neurons in the hidden layers of encoder and decoder.
            device: 'cpu' or 'cuda
        """
        super(VAE, self).__init__()
        
        # Set device
        self.device = device
        
        # TODO: Set dimensions: input dim, latent dim, and no. of neurons in the hidden layer

        # TODO: Initialize the encoder using nn.Sequential with appropriate layer dimensions, types (linear, ReLu, Sigmoid etc.).

        # TODO: Initialize a linear layer for computing the mean (one of the outputs of the encoder)
        
        # TODO: Initialize a linear layer for computing the variance (one of the outputs of the encoder)

        # TODO: Initialize the decoder using nn.Sequential with appropriate layer dimensions, types (linear, ReLu, Sigmoid etc.).



    def encode_data(self, x:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """ Forward pass throguh the encoder. 

        Args:
            x (torch.Tensor): Input data

        Returns:
            tuple[torch.Tensor, torch.Tensor]: mean, log of variance
        """
        # TODO: Implement method!!
        pass

    def reparameterize(self, mu:torch.Tensor, logvar:torch.Tensor) -> torch.Tensor:
        """ Use the reparameterization trick for sampling from a Gaussian distribution.

        Args:
            mu (torch.Tensor): Mean of the Gaussian distribution.
            logvar (torch.Tensor): Log variance of the Gaussian distribution.

        Returns:
            torch.Tensor: Sampled latent vector.
        """
        # TODO: Implement method!!
        pass

    def decode_data(self, z:torch.Tensor) -> torch.Tensor:
        """ Decode latent vectors to reconstruct data.

        Args:
            z (torch.Tensor): Latent vector.

        Returns:
            torch.Tensor: Reconstructed data.
        """
        # TODO: Implement method!!
        pass


    def generate_data(self, num_samples:int) -> torch.Tensor:
        """ Generate data by sampling and decoding 'num_samples' vectors in the latent space.

        Args:
            num_samples (int): Number of generated data samples.

        Returns:
            torch.Tensor: generated samples.
        """
        # TODO: Implement method!!
        # Hint (You may need to use .to(self.device) for sampling the latent vector!)
        pass
    
    def forward(self, x:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Forward pass of the VAE.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: reconstructed data, mean of gaussian distribution (encoder), variance of gaussian distribution (encoder)
        """
        # TODO: Implement method!!
        pass




