import pandas
import torch
import numpy as np
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torchvision.transforms import v2
from torch.utils.data import DataLoader

from main import VAE


LATENT_DIM = 20 
DEVICE = "cpu" # cuda"
BATCH_SIZE = 128
SEED = 0
SAVE_PLOTS = False 
kl_betas = [0.0, 0.1, 0.4, 0.6, 0.9, 1.0, 2.0, 10.0]

prior = torch.distributions.Normal(torch.zeros(LATENT_DIM).to(DEVICE), torch.ones(LATENT_DIM).to(DEVICE))

train_ds = datasets.MNIST(root="data", train=True, download=True, transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]))
test_ds = datasets.MNIST(root="data", train=False, download=True, transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]))

X_train = train_ds.data.float().div(255).view(-1, 784).to(DEVICE)
y_train = train_ds.targets
X_test = test_ds.data.float().div(255).view(-1, 784).to(DEVICE)
y_test = test_ds.targets

vaes, recon_losses, kl_losses = [], {}, {} 

for beta in kl_betas:
    vae = VAE(784, LATENT_DIM, True).to(DEVICE)
    vae.load_state_dict(torch.load(f"./models/vae_beta={beta}.pt"))
    vaes.append([vae, beta])
    recon_losses = torch.load(f"./models/vae_recon_losses_beta_comparison.pkl")
    kl_losses = torch.load(f"./models/vae_kl_losses_beta_comparison.pkl")



# loss comparison
for beta in kl_betas:
    plt.plot(recon_losses[beta], label=f"β={beta}")
plt.legend()
plt.title("Reconstruction loss")
plt.show()

for beta in kl_betas:
    plt.plot(kl_losses[beta], label=f"β={beta}")
plt.legend()
plt.title("KL loss")
plt.show()



# sampling from prior comparison
torch.manual_seed(SEED)
N_SAMPLES = 10
fig, axs = plt.subplots(len(vaes), N_SAMPLES, figsize=(12, 1.3 * len(vaes)))
samples = prior.sample((N_SAMPLES,))

for vae_idx, [vae, beta] in enumerate(vaes):

    with torch.no_grad():
        vae_decoded = vae.decode(samples).mean.cpu()
    
    for i in range(N_SAMPLES):
        ax = axs[vae_idx, i]
        ax.imshow(vae_decoded[i].view(28, 28), cmap="gray")
        ax.set_xticks([]); ax.set_yticks([])

    axs[vae_idx, 0].set_ylabel(f"β={beta}", rotation=0, ha="right", va="center")

fig.suptitle("decoded samples from N(0, I)")
if SAVE_PLOTS: plt.savefig("plots/beta_prior_samples.png", dpi=500, bbox_inches="tight")
plt.show()



# reconstructing digits comparison
idxs = [(y_test == d).nonzero()[0].item() for d in range(10)]
imgs = X_test[idxs]
torch.manual_seed(0)
imgs_bin = torch.bernoulli(imgs)

fig, axs = plt.subplots(1 + len(vaes), 10, figsize=(12, 1.3 * (1 + len(vaes))))
for d in range(10):
    axs[0, d].imshow(imgs[d].view(28, 28).cpu(), cmap="gray")
    axs[0, d].set_title(d)
axs[0, 0].set_ylabel("Real", rotation=0, ha="right", va="center")

for vae_idx, (vae, beta) in enumerate(vaes):
    with torch.no_grad():
        z_vae = vae.encode(imgs_bin).mean
        recon_vae = vae.decode(z_vae).probs.cpu()
    for d in range(10):
        axs[vae_idx + 1, d].imshow(recon_vae[d].view(28, 28), cmap="gray")
    axs[vae_idx + 1, 0].set_ylabel(f"β={beta}", rotation=0, ha="right", va="center")

for ax in axs.flat: ax.set_xticks([]); ax.set_yticks([])
plt.show()
