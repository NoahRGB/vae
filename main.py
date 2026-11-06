import pandas
import torch
import numpy as np
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torchvision.transforms import v2
from torch.utils.data import DataLoader

class VAE(torch.nn.Module):
    def __init__(self, in_features, latent_dim, variational: bool):
        super(VAE, self).__init__()
        self.variational = variational

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(in_features, 500),
            torch.nn.Tanh(),
        )
        self.enc_mu = torch.nn.Linear(500, latent_dim)
        self.enc_log_std = torch.nn.Linear(500, latent_dim)

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 500),
            torch.nn.Tanh(),
        )
        self.decoder_logits = torch.nn.Linear(500, in_features)


    def encode(self, inp):
        encoder_out = self.encoder(inp)
        if self.variational:
            mu_out = self.enc_mu(encoder_out)
            std_out = self.enc_log_std(encoder_out).exp()
            return torch.distributions.Normal(mu_out, std_out)
        return self.enc_mu(encoder_out)
    
    def decode(self, inp):
        decoder_out = self.decoder(inp)
        return torch.distributions.Bernoulli(logits=self.decoder_logits(decoder_out))

def train(epochs, batch_size, data, model, optim, device):

    recon_losses = []
    kl_losses = []
    latent_dim = model.enc_mu.out_features
    prior = torch.distributions.Normal(torch.zeros(latent_dim).to(device), torch.ones(latent_dim).to(device))

    for epoch in range(epochs):
    
        random_perm = torch.randperm(data.shape[0]).to(device)
        for mb in range(0, data.shape[0], batch_size):
    
            mb_idxs = random_perm[mb : mb + batch_size]
    
            feature = torch.bernoulli(data[mb_idxs])
            
            if model.variational:
                latent_dist = model.encode(feature)
                latent = latent_dist.rsample()
            else:
                latent = model.encode(feature)

            recon_dist = model.decode(latent)
            recon_loss = -recon_dist.log_prob(feature).sum(-1)
            kl_loss = torch.distributions.kl_divergence(latent_dist, prior).sum(-1) if model.variational else torch.zeros_like(recon_loss)

            loss = (recon_loss + kl_loss).sum()
    
            optim.zero_grad()
            loss.backward()
            optim.step()
            recon_losses.append(recon_loss.mean().item())
            kl_losses.append(kl_loss.mean().item())
    
        print(f"epoch {epoch+1}/{epochs} complete")
    return recon_losses, kl_losses 

BATCH_SIZE = 128
LATENT_DIM = 30 
LR = 0.001
EPOCHS = 100
DEVICE = "cpu" # cuda"
SAVE_PLOTS = True 

train_ds = datasets.MNIST(root="data", train=True, download=True, transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]))
test_ds = datasets.MNIST(root="data", train=False, download=True, transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]))

X_train = train_ds.data.float().div(255).view(-1, 784).to(DEVICE)
y_train = train_ds.targets
X_test = test_ds.data.float().div(255).view(-1, 784).to(DEVICE)
y_test = test_ds.targets

batches_per_epoch = np.ceil(len(X_train) / BATCH_SIZE)
training_samples = EPOCHS * len(X_train)

vae = VAE(784, LATENT_DIM, True).to(DEVICE)
vae_optim = torch.optim.Adam(vae.parameters(), lr=LR)

ae = VAE(784, LATENT_DIM, False).to(DEVICE)
ae_optim = torch.optim.Adam(ae.parameters(), lr=LR)


print(f"epochs = {EPOCHS}, batch size = {BATCH_SIZE}, lr = {LR}, Nz = {LATENT_DIM}")
print(f"each epoch contains {len(X_train)} samples in {batches_per_epoch} batches")
print(f"total training samples: {training_samples}\n")

print("training VAE")
vae_recon_losses, vae_kl_losses = train(EPOCHS, BATCH_SIZE, X_train, vae, vae_optim, DEVICE)

print("\ntraining AE")
ae_recon_losses, ae_kl_losses = train(EPOCHS, BATCH_SIZE, X_train, ae, ae_optim, DEVICE)


# plot VAE/AE losses
fig, axs = plt.subplots(2)
axs[0].plot(vae_recon_losses, label="VAE recon")
axs[0].plot(ae_recon_losses, label="AE recon")
axs[1].plot(vae_kl_losses, label="VAE KL")
axs[1].plot(ae_kl_losses, label="AE KL")
axs[0].legend()
axs[1].legend()
fig.suptitle(f"batch size = {BATCH_SIZE}, epochs={EPOCHS}, lr={LR}, Nz={LATENT_DIM}, training samples={training_samples}")
if SAVE_PLOTS: plt.savefig("plots/losses.png", dpi=500, bbox_inches="tight")
plt.show()


# plot real/VAE decoded/AE decoded image samples
with torch.no_grad():
    # get test images of labels from 0-9
    idxs = [(y_test == d).nonzero()[0].item() for d in range(10)]
    imgs = X_test[idxs]
    
    # encode + decode
    z_vae = vae.encode(torch.bernoulli(imgs)).mean
    z_ae = ae.encode(torch.bernoulli(imgs))

    recon_vae = vae.decode(z_vae).probs
    recon_ae = ae.decode(z_ae).probs
    
    # display real image next to decoded latent per example
    fig, axs = plt.subplots(3, 10, figsize=(12, 6))
    for d in range(10):
        axs[0, d].imshow(imgs[d].view(28, 28).cpu(), cmap="gray")
        axs[1, d].imshow(recon_vae[d].view(28, 28).cpu(), cmap="gray")
        axs[2, d].imshow(recon_ae[d].view(28, 28).cpu(), cmap="gray")
        axs[0, d].set_title(d)
        axs[0, d].set_xticks([]); axs[0, d].set_yticks([]);
        axs[1, d].set_xticks([]); axs[1, d].set_yticks([]);
        axs[2, d].set_xticks([]); axs[2, d].set_yticks([]);
    axs[0, 0].set_ylabel("Real")
    axs[1, 0].set_ylabel("VAE decoded")
    axs[2, 0].set_ylabel("AE decoded")
    fig.suptitle(f"batch size = {BATCH_SIZE}, epochs={EPOCHS}, lr={LR}, Nz={LATENT_DIM}, training samples={training_samples}")
    if SAVE_PLOTS: plt.savefig("plots/recons.png", dpi=500, bbox_inches="tight")
    plt.show()



# plot latent space as X/Y scatterplot (if Nz > 2, project onto top 2 principal components)
def pca_2d(z):
    """project (N, D) latents onto their top 2 principal components"""
    z = z - z.mean(0)
    _, S, Vh = torch.linalg.svd(z, full_matrices=False)
    explained = (S**2 / (S**2).sum())[:2]
    return z @ Vh[:2].T, explained

with torch.no_grad():
    mu_vae = vae.encode(torch.bernoulli(X_test)).mean.cpu()
    z_ae = ae.encode(torch.bernoulli(X_test)).cpu()

fig, axs = plt.subplots(1, 2, figsize=(13, 5))
for ax, z, name in [(axs[0], mu_vae, "VAE"), (axs[1], z_ae, "AE")]:

    # if Nz > 2, do PCA to get 2 dims
    # otherwise just plot z1/z2 as X/Y
    if LATENT_DIM > 2:
        z, ev = pca_2d(z)
        ax.set_xlabel(f"PC1 ({ev[0]:.0%} var)"); ax.set_ylabel(f"PC2 ({ev[1]:.0%} var)")
    else:
        ax.set_xlabel("z1"); ax.set_ylabel("z2")

    sc = ax.scatter(z[:, 0], z[:, 1], c=y_test, cmap="tab10", s=2, alpha=0.6)
    ax.set_title(name)
    fig.colorbar(sc, ax=ax, ticks=range(10))
plt.tight_layout()
fig.suptitle(f"batch size = {BATCH_SIZE}, epochs={EPOCHS}, lr={LR}, Nz={LATENT_DIM}, training samples={training_samples}", y=1.02)
if SAVE_PLOTS: plt.savefig("plots/2dim_plot.png", dpi=500, bbox_inches="tight")
plt.show()
