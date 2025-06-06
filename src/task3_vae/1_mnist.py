import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from model import VAE
from utils import *

""" 
This script is used to train and test the VAE.
"""

############################################################
## Subtasks 3.3 & 3.4 in the worksheet ##
############################################################
# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(0)
torch.manual_seed(0)

# Load the MNIST dataset: Train and test
train_dataset = MNIST(".", train=True, download=True, transform=ToTensor())
test_dataset = MNIST(".", train=False, download=True, transform=ToTensor())


# Set the learning rate, batch size and no. of epochs
learning_rate = 0.001
batch_size = 128
epochs = 50

# Create an instance of Dataloader for train_dataset using torch.utils.data, use appropriate batch size, keep shuffle=True.
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Create an instance of Dataloader for test_dataset using torch.utils.data, use appropriate batch size, keep shuffle=False.
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Set dimensions: input dim, latent dim, and no. of neurons in the hidden layer
d_in = 784
d_latent = 2
d_hidden_layer = 256

# Instantiate the VAE model with a latent dimension of 2, using the utility function instantiate_vae() from utils
vae = instantiate_vae(d_in, d_latent, d_hidden_layer, device)

# Set up an appropriate optimizer from torch.optim with an appropriate learning rate
optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

plots_at_epochs = [0, 4, 24, epochs - 1]  # generate plots at epoch numbers

# Compute train and test losses by performing the training loop using the utility function training_loop() from utils
train_losses, test_losses = training_loop(
    vae, optimizer, train_loader, test_loader, epochs, plots_at_epochs, device
)

# Plot the loss curve using the utility function plot_loss() from utils
plot_loss(train_losses, test_losses)


##############################################################
##### Subtask 3.5 in the worksheet #######
##############################################################
# Create the VAE model with a latent dimension of 32
d_latent_32 = 32
# Instantiate the VAE model with a latent dimension of 32, using the utility function instantiate_vae() from utils
vae_32 = instantiate_vae(d_in, d_latent_32, d_hidden_layer, device)

# Set up an appropriate optimizer from torch.optim with an appropriate learning rate
optimizer_32 = optim.Adam(vae_32.parameters(), lr=learning_rate)

# Compute train and test losses by performing the training loop using the utility function training_loop() from utils
train_losses_32, test_losses_32 = training_loop(
    vae_32,
    optimizer_32,
    train_loader,
    test_loader,
    epochs,
    plots_at_epochs,
    device,
)

# (5a) Compare 15 generated digits using the utility function reconstruct_digits()
reconstruct_digits(vae_32, test_loader, device, num_digits=15)
generate_digits(vae_32, num_samples=15)

# (5b) Plot the loss curve using the utility function plot_loss()
plot_loss(train_losses_32, test_losses_32)
