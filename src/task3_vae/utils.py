import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from model import VAE
import numpy.typing as npt

# Define a loss function that combines binary cross-entropy and Kullback-Leibler divergence
def reconstruction_loss(x_reconstructed:torch.Tensor, x:torch.Tensor) -> torch.Tensor:
    """Compute the reconstruction loss.

    Args:
        x_reconstructed (torch.Tensor): Reconstructed data
        x (torch.Tensor): raw/original data

    Returns:
        (torch.Tensor): reconstruction loss
    """
    # TODO: Implement method! 
    mse = F.mse_loss(x_reconstructed, x, reduction='sum')
    raise NotImplementedError("mse: ", mse)
    return mse

def kl_loss(logvar:torch.Tensor, mu:torch.Tensor) -> torch.Tensor:
    """ Compute the Kullback-Leibler (KL) divergence loss using the encoded data into the mean and log-variance.

    Args:
        logvar (torch.Tensor): log of variance (from the output of the encoder)
        mu (torch.Tensor): mean (from the output of the encoder)

    Returns:
        (torch.Tensor): KL loss
    """
    # TODO: Implement method! 
    kl = -0.5 * torch.sum(1.0 + logvar - mu**2 - logvar.exp())
    raise NotImplementedError("kl: ", kl)
    return kl

# Function to compute ELBO loss
def elbo_loss(x:torch.Tensor, x_reconstructed:torch.Tensor, mu:torch.Tensor, logvar:torch.Tensor):
    """Compute Evidence Lower BOund (ELBO) Loss by combining the KL loss and reconstruction loss. 

    Args:
        x (torch.Tensor): raw/original data
        x_reconstructed (torch.Tensor): Reconstructed data
        mu (torch.Tensor): mean (from the output of the encoder)
        logvar (torch.Tensor): log of variance (from the output of the encoder)

    Returns:
        (torch.Tensor): ELBO loss
    """
    # TODO: Implement method! Hint(You may need to reshape x using x.view(. , .)!)
    
    x = x.view(x_reconstructed.shape)
    
    reconstruction = reconstruction_loss(x_reconstructed, x)
    
    # KL divergence loss
    kl = kl_loss(logvar, mu)
    

    return reconstruction + kl

# Function for training the VAE
def train_epoch(model:object, optimizer:object, dataloader:object, device) -> np.float64:
    """ Train the vae for one epoch and return the training loss on the epoch. 

    Args:
        model (object): The model (of class VAE)
        optimizer (object): Adam optimizer (from torch.optim)
        dataloader (object): Data loader combines a dataset and a sampler, and provides an iterable over the given dataset (from torch.utils.data).
        device: The device (e.g., 'cuda' or 'cpu') on which the training is to be done.

    Returns:
        np.float64: training loss
    """
    model.train()
    total_loss = 0
    for data, _ in dataloader:
        data = data.view(-1, int(np.shape(data)[-1] *np.shape(data)[-2])).to(device)
        
        # TODO: Set gradient to zero! You can use optimizer.zero_grad()!

        # TODO: Perform forward pass of the VAE
        
        # TODO: Compute ELBO loss

        # TODO: Compute gradients 

        # TODO: Perform an optimization step

        # TODO: Compute total_loss and return the total_loss/len(dataloader.dataset)

    pass

def evaluate(model:object, dataloader:object, device)-> np.float64:
    """ Evaluate the model on the test data and return the test loss.

    Args:
        model (object): The model (of class VAE)
        dataloader (object): Data loader combines a dataset and a sampler, and provides an iterable over the given dataset (from torch.utils.data).
        device: The device (e.g., 'cuda' or 'cpu').

    Returns:
        np.float64: test loss.
    """
    # TODO: Implement method! 
    # Hint: Do not forget to deactivate the gradient calculation!
    # return total_loss/len(dataloader.dataset)
    pass

def latent_representation(model:object, dataloader:object, device) -> None:
    """Plot the latent representation of the data.

    Args:
        model (object): The model (of class VAE).
        dataloader (object): Data loader combines a dataset and a sampler, and provides an iterable over the given dataset (from torch.utils.data).
        device: The device (e.g., 'cuda' or 'cpu').
    """
    # TODO: Implement method! 
    # Hint: Do not forget to deactivate the gradient calculation!
    pass

# Function to plot reconstructed digits
def reconstruct_digits(model:object, dataloader:object, device, num_digits:int =15) -> None:
    """ Plot reconstructed digits. 

    Args:
        model (object): The model (of class VAE).
        dataloader (object): Data loader combines a dataset and a sampler, and provides an iterable over the given dataset (from torch.utils.data).
        device: The device (e.g., 'cuda' or 'cpu').
        num_digits (int, optional): No. of digits to be re-constructed. Defaults to 15.
    """
    # TODO: Implement method! 
    # Hint: Do not forget to deactivate the gradient calculation!
    pass


# Function to plot generated digits
def generate_digits(model:object, num_samples:int =15) -> None:
    """ Generate 'num_samples' digits.

    Args:
        model (object): The model (of class VAE).
        num_samples (int, optional): No. of samples to be generated. Defaults to 15.
    """
    # TODO: Implement method! 
    # Hint: Do not forget to deactivate the gradient calculation!
    pass

# Function to plot the loss curve
def plot_loss(train_losses, test_losses):
    epochs = len(train_losses)
    plt.figure()
    plt.plot(range(1, epochs+1), train_losses, label='Train')
    plt.plot(range(1, epochs+1), test_losses, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('ELBO Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.show()
    plt.savefig('loss_curve.png')


def training_loop(vae:object, optimizer:object, train_loader:object, test_loader:object, epochs:int, plots_at_epochs:list, device) -> tuple [list, list]:
    """ Train the vae model. 

    Args:
        vae (object): The model (of class VAE).
        optimizer (object): Adam optimizer (from torch.optim).
        train_loader (object): A data loader that combines the training dataset and a sampler, and provides an iterable over the given dataset (from torch.utils.data).
        test_loader (object): A data loader that combines the test dataset and a sampler, and provides an iterable over the given dataset (from torch.utils.data).
        epochs (int): No. of epochs to train the model.
        plots_at_epochs (list): List of integers containing epoch numbers at which the plots are to be made.
        device: The device (e.g., 'cuda' or 'cpu').

    Returns:
        tuple [list, list]: Lists train_losses, test_losses containing train and test losses at each epoch.
    """
    # Lists to store the training and test losses
    train_losses = []
    test_losses = []
    for epoch in range(epochs):
        # TODO: Compute training loss for one epoch
        
        # TODO: Evaluate loss on the test dataset
        
        # TODO: Append train and test losses to the lists train_losses and test_losses respectively

        print(f'Epoch , Train Loss: , Test Loss: ', )

        # TODO: For specific epoch numbers described in the worksheet, plot latent representation, reconstructed digits, generated digits after specific epochs


    # TODO: return train_losses, test_losses
    pass


def instantiate_vae(d_in, d_latent, d_hidden_layer, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """Instantiate the variational autoencoder.

    Args:
        d_in (int): Input dimension.
        d_latent (int): Latent dimension.
        d_hidden_layer (int): Number of neurons in each hidden layer of the encoder and decoder.
        device: e.g., 'cuda' or 'cpu'. Defaults to torch.device('cuda' if torch.cuda.is_available() else 'cpu').

    Returns:
        object: An object of class VAE
    """
    return VAE(d_in, d_latent, d_hidden_layer, device).to(device)