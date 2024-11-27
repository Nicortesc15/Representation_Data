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
    bce_loss = F.binary_cross_entropy(x_reconstructed, x, reduction='sum')
    return bce_loss


def kl_loss(logvar:torch.Tensor, mu:torch.Tensor) -> torch.Tensor:
    """ Compute the Kullback-Leibler (KL) divergence loss using the encoded data into the mean and log-variance.

    Args:
        logvar (torch.Tensor): log of variance (from the output of the encoder)
        mu (torch.Tensor): mean (from the output of the encoder)

    Returns:
        (torch.Tensor): KL loss
    """
    kl = -0.5 * torch.sum(1.0 + logvar - mu**2 - logvar.exp())
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
        
        # Set gradient to zero
        optimizer.zero_grad()

        # Perform forward pass of the VAE
        data_reconstructed, mu, logvar = model(data)

        # Compute ELBO loss
        loss = elbo_loss(data, data_reconstructed, mu, logvar)

        # Compute gradients 
        loss.backward()

        # Perform an optimization step
        optimizer.step()

        # Compute total_loss and return the total_loss/len(dataloader.dataset)
        total_loss += loss.item()
    
    return total_loss / len(dataloader.dataset)


def evaluate(model:object, dataloader:object, device)-> np.float64:
    """ Evaluate the model on the test data and return the test loss.

    Args:
        model (object): The model (of class VAE)
        dataloader (object): Data loader combines a dataset and a sampler, and provides an iterable over the given dataset (from torch.utils.data).
        device: The device (e.g., 'cuda' or 'cpu').

    Returns:
        np.float64: test loss.
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, _ in dataloader:
            data = data.view(-1, int(np.shape(data)[-1] *np.shape(data)[-2])).to(device)
            
            # Perform forward pass of the VAE
            x_reconstructed, mu, logvar = model(data)

            # Compute ELBO loss
            loss = elbo_loss(data, x_reconstructed, mu, logvar)

            # Compute total_loss and return the total_loss/len(dataloader.dataset)
            total_loss += loss.item()
    
    return total_loss / len(dataloader.dataset)


def latent_representation(model:object, dataloader:object, device) -> None:
    """Plot the latent representation of the data.

    Args:
        model (object): The model (of class VAE).
        dataloader (object): Data loader combines a dataset and a sampler, and provides an iterable over the given dataset (from torch.utils.data).
        device: The device (e.g., 'cuda' or 'cpu').
    """
    model.eval()
    latents, labels = [], []
    with torch.no_grad():
        for data, target in dataloader:
            data = data.view(-1, int(np.shape(data)[-1] *np.shape(data)[-2])).to(device)            
            mu, logvar = model.encode_data(data)                                                
            latents.append(mu.cpu())                                                            
            labels.append(target)

    latents = torch.cat(latents)
    labels = torch.cat(labels)

    # Plot latent space
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(latents[:, 0], latents[:, 1], c=labels, cmap='tab10', s=2)
    plt.colorbar(scatter, label='Class Label')
    plt.title('Latent Representation')
    plt.xlabel('$z_1$')
    plt.ylabel('$z_2$')
    plt.show()


# Function to plot reconstructed digits
def reconstruct_digits(model:object, dataloader:object, device, num_digits:int =15) -> None:
    """ Plot reconstructed digits. 

    Args:
        model (object): The model (of class VAE).
        dataloader (object): Data loader combines a dataset and a sampler, and provides an iterable over the given dataset (from torch.utils.data).
        device: The device (e.g., 'cuda' or 'cpu').
        num_digits (int, optional): No. of digits to be re-constructed. Defaults to 15.
    """
    model.eval()
    with torch.no_grad():
        # Extract the first num_digits of the next batch and flatten the dimension
        data, label = next(iter(dataloader))
        data = data[:num_digits].view(num_digits, -1).to(device)
        reconstructed, mu, logvar = model(data)

        # Reshape for visualization
        data = data.cpu().view(-1, 28, 28)
        reconstructed = reconstructed.cpu().view(-1, 28, 28)

        # Plot original and reconstructed images
        fig, axes = plt.subplots(2, num_digits, figsize=(15, 4))
        for i in range(num_digits):
            axes[0, i].imshow(data[i], cmap='gray')
            axes[0, i].axis('off')
            axes[1, i].imshow(reconstructed[i], cmap='gray')
            axes[1, i].axis('off')

        fig.text(0.5, 0.9, 'Original Digits', ha='center', va='center', fontsize=12)
        fig.text(0.5, 0.5, 'Reconstructed Digits', ha='center', va='center', fontsize=12)
        plt.show()


# Function to plot generated digits
def generate_digits(model:object, num_samples:int =15) -> None:
    """ Generate 'num_samples' digits.

    Args:
        model (object): The model (of class VAE).
        num_samples (int, optional): No. of samples to be generated. Defaults to 15.
    """
    model.eval()
    with torch.no_grad():
        generated = model.generate_data(num_samples).cpu().view(-1, 28, 28)

        # Plot generated images
        fig, axes = plt.subplots(1, num_samples, figsize=(15, 4))
        for i in range(num_samples):
            axes[i].imshow(generated[i], cmap='gray')
            axes[i].axis('off')
        plt.suptitle('Generated Digits')
        plt.show()


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
        # Compute training loss for one epoch
        train_loss = train_epoch(vae, optimizer, train_loader, device)

        # Evaluate loss on the test dataset
        test_loss = evaluate(vae, test_loader, device)

        # Append train and test losses to the lists train_losses and test_losses respectively
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        print(f"Epoch {epoch}, Train Loss: {train_loss:.5f}, Test Loss: {test_loss:.5f}")

        # For specific epoch numbers described in the worksheet, plot latent representation, reconstructed digits, generated digits after specific epochs
        if vae.d_latent == 2:
            if epoch in plots_at_epochs:
                print(f"=== Plots after Epoch: {epoch} ===")
                # Plot latent representation
                print(f"1. Plot Latent Representation")
                latent_representation(vae, test_loader, device)

                # Plot reconstructed digits
                print(f"2. Plot Original and Reconstructed Digits")
                reconstruct_digits(vae, test_loader, device, num_digits=15)

                # Plot generated digits
                print(f"3. Plot Generated Digits")
                generate_digits(vae, num_samples=15)

    return train_losses, test_losses


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

