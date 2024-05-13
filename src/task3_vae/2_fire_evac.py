import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from model import VAE
from utils import * 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

""" This script is used to train and test the VAE on the fire_evac dataset. 
(Bonus: You can simulate trajectories with Vadere, for bonus points.) 
Not included in automated tests, as it's open-ended.
"""

# TODO: Download the FireEvac dataset 

# TODO: Make a scatter plot to visualise it.

# TODO: Train a VAE on the FireEvac data

# TODO: Make a scatter plot of the reconstructed test set

# TODO: Make a scatter plot of 1000 generated samples.

# TODO: Generate data to estimate the critical number of people for the MI building

