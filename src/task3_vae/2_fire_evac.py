import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from utils_fire import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" 
This script is used to train and test the VAE on the fire_evac dataset. 
(Bonus: You can simulate trajectories with Vadere, for bonus points.) 
Not included in automated tests, as it's open-ended.
"""

np.random.seed(0)
torch.manual_seed(0)

# Download the FireEvac dataset
train_dataset = np.load("data/FireEvac_train_set.npy")
test_dataset = np.load("data/FireEvac_test_set.npy")

# Rescale data
train_scaled, test_scaled, scaling_data_min, scaling_data_max = rescale_data(
    train_dataset, test_dataset
)

# Make a scatter plot to visualise it.
plt.scatter(
    train_scaled[:, 0],
    train_scaled[:, 1],
    alpha=0.5,
    label="Training Data",
    s=5,
)
plt.scatter(
    test_scaled[:, 0], test_scaled[:, 1], alpha=0.5, label="Test Data", s=5
)
plt.title("FireEvac Dataset")
plt.xlabel("x-position")
plt.ylabel("y-position")
plt.legend()
plt.show()

# Train a VAE on the FireEvac data
train_d = FireDataset(train_scaled)
test_d = FireDataset(test_scaled)

# Set the learning rate, batch size and no. of epochs
learning_rate = 0.0001
batch_size = 64
epochs = 200

# Create an instance of Dataloader for train_dataset using torch.utils.data, use appropriate batch size, keep shuffle=True.
train_loader = DataLoader(train_d, batch_size=batch_size, shuffle=True)

# Create an instance of Dataloader for test_dataset using torch.utils.data, use appropriate batch size, keep shuffle=False.
test_loader = DataLoader(test_d, batch_size=batch_size, shuffle=False)


# Set dimensions: input dim, latent dim, and no. of neurons in the hidden layer
d_in = 2
d_latent = 2
d_hidden_layer = 32

# Instantiate the VAE model with a latent dimension of 2, using the utility function instantiate_vae() from utils
vae = instantiate_vae(d_in, d_latent, d_hidden_layer, device)

# Set up an appropriate optimizer from torch.optim with an appropriate learning rate
optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

plots_at_epochs = [50, 100, epochs - 1]  # generate plots at epoch numbers

# Compute train and test losses by performing the training loop using the utility function training_loop() from utils
train_losses, test_losses = training_loop(
    vae, optimizer, train_loader, test_loader, epochs, plots_at_epochs, device
)

# Plot the loss curve using the utility function plot_loss() from utils
plot_loss(train_losses, test_losses)

# Generate data to estimate the critical number of people for the MI building
# x,y coordinates of the corners of the orange rectangle after rescaling
orange_rect_corner_1 = [0.68521264, 0.63112379]
orange_rect_corner_2 = [0.79284455, 0.43787233]

# Rectangle bounds
x_min, x_max = orange_rect_corner_1[0], orange_rect_corner_2[0]
y_min, y_max = orange_rect_corner_2[1], orange_rect_corner_1[1]

# Initialize counters
person_count = 0
person_in_rect = 0

# Generate positions until 100 are in the orange rectangle
people_added = 100
max_attempts = 100000

while person_in_rect < 100:
    # Generate a batch of positions
    positions = generate_positions(vae, num_samples=people_added)
    person_count += people_added

    # Check which positions are inside the rectangle
    in_rect = (
        (x_min <= positions[:, 0])
        & (positions[:, 0] <= x_max)
        & (y_min <= positions[:, 1])
        & (positions[:, 1] <= y_max)
    )

    # Count new persons in the rectangle
    person_in_rect += np.sum(in_rect)

    # Stop if the maximum number of attempts is exceeded
    if person_count > max_attempts:
        print(
            "Unable to estimate the critical number of people for the MI building. Please check your model."
        )
        break

# Output the result
if person_in_rect >= 100:
    print(
        "Estimated critical number of people for the MI building: ",
        person_count,
    )
