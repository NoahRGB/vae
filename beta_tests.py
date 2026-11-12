import pandas
import torch
import numpy as np
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torchvision.transforms import v2
from torch.utils.data import DataLoader

from main import VAE, train


BATCH_SIZE = 128
LATENT_DIM = 20 
LR = 0.001
EPOCHS = 100
DEVICE = "cuda" # cuda"
SAVE_PLOTS = True 
SEED = 0


prior = torch.distributions.Normal(torch.zeros(LATENT_DIM).to(DEVICE), torch.ones(LATENT_DIM).to(DEVICE))

train_ds = datasets.MNIST(root="data", train=True, download=True, transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]))
test_ds = datasets.MNIST(root="data", train=False, download=True, transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]))

X_train = train_ds.data.float().div(255).view(-1, 784).to(DEVICE)
y_train = train_ds.targets
X_test = test_ds.data.float().div(255).view(-1, 784).to(DEVICE)
y_test = test_ds.targets

batches_per_epoch = np.ceil(len(X_train) / BATCH_SIZE)
training_samples = EPOCHS * len(X_train)


recon_losses, kl_losses = {}, {}
kl_betas = [0.0, 0.1, 0.4, 0.6, 0.9, 1.0, 2.0, 10.0]

for kl_beta in kl_betas:

    print(f"\ntraining beta={kl_beta}")
    torch.manual_seed(SEED)

    vae = VAE(784, LATENT_DIM, True).to(DEVICE)
    vae_optim = torch.optim.Adam(vae.parameters(), lr=LR)

    vae_recon_losses, vae_kl_losses = train(EPOCHS, BATCH_SIZE, X_train, vae, vae_optim, kl_beta, prior, DEVICE)

    recon_losses[kl_beta] = vae_recon_losses
    kl_losses[kl_beta] = vae_kl_losses
    torch.save(vae.state_dict(), f"./models/vae_beta={kl_beta}.pt")

torch.save(recon_losses, f"./models/vae_recon_losses_beta_comparison.pkl")
torch.save(kl_losses, f"./models/vae_kl_losses_beta_comparison.pkl")

