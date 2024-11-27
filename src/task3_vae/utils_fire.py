import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from model_fire import VAE
import numpy.typing as npt
from torch.utils.data import Dataset

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

    x = x.view(-1, x.shape[-1])

    mse_loss = F.mse_loss(x_reconstructed, x, reduction='sum')
    return mse_loss

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

    # Weighting variable
    weight = 0.01
    return (1 - weight) * reconstruction + weight * kl

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
        data = data.view(data.shape[0], -1).to(device)
        
        # TODO: Set gradient to zero! You can use optimizer.zero_grad()!
        optimizer.zero_grad()
        # TODO: Perform forward pass of the VAE
        x_reconstructed, mu, logvar = model(data)
        # TODO: Compute ELBO loss
        loss = elbo_loss(data, x_reconstructed, mu, logvar)
        # TODO: Compute gradients 
        loss.backward()
        # TODO: Perform an optimization step
        optimizer.step()
        # TODO: Compute total_loss and return the total_loss/len(dataloader.dataset)
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
    # TODO: Implement method! 
    # Hint: Do not forget to deactivate the gradient calculation!
    # return total_loss/len(dataloader.dataset)
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, _ in dataloader:
            data = data.view(data.shape[0], -1).to(device)
            
            # TODO: Perform forward pass of the VAE
            x_reconstructed, mu, logvar = model(data)
            # TODO: Compute ELBO loss
            loss = elbo_loss(data, x_reconstructed, mu, logvar)
            # TODO: Compute total_loss and return the total_loss/len(dataloader.dataset)
            total_loss += loss.item()
    
    return total_loss / len(dataloader.dataset)

def latent_representation(model:object, dataloader:object, device) -> None:
    """Plot the latent representation of the data.

    Args:
        model (object): The model (of class VAE).
        dataloader (object): Data loader combines a dataset and a sampler, and provides an iterable over the given dataset (from torch.utils.data).
        device: The device (e.g., 'cuda' or 'cpu').
    """
    # TODO: Implement method! 
    # Hint: Do not forget to deactivate the gradient calculation!
    model.eval()
    latents = []
    with torch.no_grad():
        for data, _ in dataloader:
            data = data.view(data.shape[0], -1).to(device)            
            mu, logvar = model.encode_data(data)                                                
            latents.append(mu.cpu().numpy())
        latents = np.concatenate(latents, axis=0)

        # Plot latent space
        plt.scatter(latents[:, 0], latents[:, 1], s = 5)
        plt.title("Latent Representation")
        plt.xlabel("z1")
        plt.ylabel("z2")
        plt.show()

# Function to plot reconstructed digits
def reconstruct_positions(model: object, dataloader: object, device, num_points: int = 15) -> None:
    """Plot original and reconstructed 2D positions for the FireEvac dataset.

    Args:
        model (object): The model (of class VAE).
        dataloader (object): Data loader providing the FireEvac dataset.
        device: The device ('cuda' or 'cpu').
        num_points (int, optional): Number of points to visualize. Defaults to 15.
    """
    model.eval()
    original_data = []
    reconstructed_data = []
    with torch.no_grad():
        
        for data, _ in dataloader:
            # Extract data from the batch and move it to the device
            data = data.view(data.shape[0], -1).to(device)
            reconstructed, _, _ = model(data)

            # Collect original and reconstructed data
            original_data.append(data.cpu().numpy())
            reconstructed_data.append(reconstructed.cpu().numpy())

        
        # Concatenate all batches into single NumPy arrays
        original_data = np.concatenate(original_data, axis=0)
        reconstructed_data = np.concatenate(reconstructed_data, axis=0)

        # Scatter plot of original and reconstructed points
        plt.figure(figsize=(8, 8))
        plt.scatter(original_data[:, 0], original_data[:, 1], color='red', alpha=0.7, label = 'Original Positions', s = 5)
        plt.scatter(reconstructed_data[:, 0], reconstructed_data[:, 1], color='blue', alpha=0.7, label = 'Reconstructed Positions', s = 5)
        plt.title('Original and Reconstructed Positions')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.show()



# Function to plot generated digits
def generate_positions(model: object, num_samples: int = 15, device='cpu') -> None:
    """Generate and visualize 'num_samples' 2D positions for the FireEvac dataset.

    Args:
        model (object): The trained VAE model.
        num_samples (int, optional): Number of samples to generate. Defaults to 15.
        device (str, optional): The device ('cuda' or 'cpu'). Defaults to 'cpu'.
    """
    model.eval()
    with torch.no_grad():
        # Generate new samples from the latent space
        generated = model.generate_data(num_samples).cpu().numpy()
    
    return generated
        

def plot_generate_positions(model: object, num_samples: int = 15, device='cpu') -> None:
    """
    Visualize 'num_samples' 2D positions for the FireEvac dataset.

    Args:
        generated (np.ndarray): Array of generated positions, shape (num_samples, 2).
        num_samples (int, optional): Number of samples to visualize. Defaults to 15.
    """

    generated = generate_positions(model, num_samples, device)
    # Scatter plot of generated samples
    plt.figure(figsize=(8, 8))
    plt.scatter(generated[:num_samples, 0], generated[:num_samples, 1], color='green', s=5)
    plt.title(f'{num_samples} Generated Positions')
    plt.xlabel('x')
    plt.ylabel('y')
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
        # TODO: Compute training loss for one epoch
        train_loss = train_epoch(vae, optimizer, train_loader, device)
        # TODO: Evaluate loss on the test dataset
        test_loss = evaluate(vae, test_loader, device) 
        # TODO: Append train and test losses to the lists train_losses and test_losses respectively
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        print(f"Epoch {epoch}, Train Loss: {train_loss:.5f}, Test Loss: {test_loss:.5f}")

        # TODO: For specific epoch numbers described in the worksheet, plot latent representation, reconstructed digits, generated digits after specific epochs
        if vae.d_latent == 2:
            if epoch in plots_at_epochs:
                print(f"=== Plots after Epoch: {epoch} ===")
                # Plot latent representation
                print(f"1. Plot Latent Representation")
                latent_representation(vae, test_loader, device)

                # Plot reconstructed positions
                print(f"2. Plot Original and Reconstructed Positions")
                reconstruct_positions(vae, test_loader, device, num_points=len(test_loader.dataset))

                # Plot generated positions
                print(f"3. Plot Generated Positions")
                plot_generate_positions(vae, num_samples=1000)

    
    # TODO: return train_losses, test_losses
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

def rescale_data(train_data: np.ndarray, test_data: np.ndarray, new_min: float = -1, new_max: float = 1) -> np.ndarray:
    """
    Rescale the data to a specified range [new_min, new_max].

    Args:
        train_data (np.ndarray): Input train data to be rescaled. Should be a NumPy array.
        test_data (np.ndarray): Input test data to be rescaled. Should be a NumPy array.
        new_min (float): The minimum value of the new range. Defaults to -1.
        new_max (float): The maximum value of the new range. Defaults to 1.

    Returns:
        np.ndarray: Rescaled data within the range [new_min, new_max].
    """
    # Put all the data together
    data = np.concatenate((train_data, test_data), axis = 0)

    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)
    data_range = data_max - data_min

    # Rescale data
    scaled_train = (train_data - data_min) / data_range  
    rescaled_train = scaled_train * (new_max - new_min) + new_min

    scaled_test = (test_data - data_min) / data_range  
    rescaled_test = scaled_test * (new_max - new_min) + new_min

    return rescaled_train, rescaled_test, data_min, data_max

class FireDataset(Dataset):
    def __init__(self, data):
        """Initialize the FireDataset. Ensure that the data is of type np.float32.

        Args:
            data (npt.NDArray[np.float64]): The data to be used for training/testing.
        """
        self.data = data.astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.data[idx]
