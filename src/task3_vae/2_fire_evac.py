import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from model_fire import VAE
from utils_fire import * 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

""" This script is used to train and test the VAE on the fire_evac dataset. 
(Bonus: You can simulate trajectories with Vadere, for bonus points.) 
Not included in automated tests, as it's open-ended.
"""

# TODO: Download the FireEvac dataset 
train_dataset = np.load('data/FireEvac_train_set.npy')
test_dataset = np.load('data/FireEvac_test_set.npy')

print(train_dataset.shape)
print(test_dataset.shape)

# TODO: Make a scatter plot to visualise it.
plt.scatter(train_dataset[:, 0], train_dataset[:, 1], alpha=0.5, label='Training Data')
plt.scatter(test_dataset[:, 0], test_dataset[:, 1], alpha=0.5, label='Test Data')
plt.title("FireEvac Dataset")
plt.xlabel("x-position")
plt.ylabel("y-position")
plt.legend()
plt.show()

# TODO: Rescale data
train_dataset_scaled = rescale_data(train_dataset)
test_dataset_scaled = rescale_data(test_dataset)

# TODO: Train a VAE on the FireEvac data

# TODO: Set the learning rate, batch size and no. of epochs
learning_rate = 0.001
batch_size = 1000     
epochs = 200 

# TODO: Create an instance of Dataloader for train_dataset using torch.utils.data, use appropriate batch size, keep shuffle=True.
train_loader = DataLoader(torch.tensor(train_dataset_scaled, dtype=torch.float32), batch_size=batch_size, shuffle=True)

# TODO: Create an instance of Dataloader for test_dataset using torch.utils.data, use appropriate batch size, keep shuffle=False.
test_loader = DataLoader(torch.tensor(test_dataset_scaled, dtype=torch.float32), batch_size=batch_size, shuffle=False)


# TODO: Set dimensions: input dim, latent dim, and no. of neurons in the hidden layer
d_in = 2
d_latent = 2
d_hidden_layer = 256

# TODO: Instantiate the VAE model with a latent dimension of 2, using the utility function instantiate_vae() from utils
vae = instantiate_vae(d_in, d_latent, d_hidden_layer, device)

# TODO. Set up an appropriate optimizer from torch.optim with an appropriate learning rate
optimizer = optim.Adam(vae.parameters(), lr = learning_rate)

plots_at_epochs = [49,99,199]  # generate plots at epoch numbers

# TODO: Compute train and test losses by performing the training loop using the utility function training_loop() from utils
train_losses, test_losses = training_loop(vae, optimizer, train_loader, test_loader, epochs, plots_at_epochs, device)

# TODO: Plot the loss curve using the utility function plot_loss() from utils
plot_loss(train_losses, test_losses)

# TODO: Make a scatter plot of the reconstructed test set

# TODO: Make a scatter plot of 1000 generated samples.

# TODO: Generate data to estimate the critical number of people for the MI building

