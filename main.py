import pandas
import torch
import numpy as np
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torchvision.transforms import v2
from torch.utils.data import DataLoader

class VAE(torch.nn.Module):
    def __init__(self, in_features, latent_dim):
        super(VAE, self).__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(in_features, 500),
            torch.nn.Tanh(),
        )
        self.enc_mu = torch.nn.Linear(500, latent_dim)
        self.enc_log_std = torch.nn.Linear(500, latent_dim)

        self.decoder = torch.nn.Sequentual(
            torch.nn.Linear(latent_dim, 500),
            torch.nn.Tanh(),
            torch.nn.Linear(500, in_features),
        )


    def encode(self, inp):
        encoder_out = self.encoder(inp)
        mu_out = self.enc_mu(encoder_out)
        std_out = self.enc_log_std(encoder_out).exp()
        return mu_out, std_out
    
    def decode(self, inp):
        return self.decoder(inp)


train_ds = datasets.MNIST(root="data", train=True, download=True, transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]))
test_ds = datasets.MNIST(root="data", train=False, download=True, transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]))

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
test_loader = DataLoader(test_ds,  batch_size=1000)

LATENT_DIM = 5
vae = VAE(784, LATENT_DIM)

for feature, label in train_loader:
    feature = feature.flatten(1) # (batch, 784)

    latent_mu, latent_std = vae.encode(feature)
    latent_dist = torch.distributions.Normal(latent_mu, latent_std)
    latent = latent_dist.rsample()

    prior = torch.distributions.Normal(torch.zeros(LATENT_DIM), torch.ones(LATENT_DIM))
