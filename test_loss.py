import torch

from dataset import loader
from model import VAE
from loss import vae_loss

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Model
model = VAE(latent_dim=128).to(device)
print("model created successfuly")

# Get one batch
images = next(iter(loader)).to(device)
print("Images shape: ",images.shape)

# Forward pass
reconstruction, mu, logvar = model(images)

# Calculalte VAE loss
total_loss, reconstruction_loss, kl_loss = vae_loss(
    reconstruction, 
    images, 
    mu, 
    logvar
)

# print losses

print("\nLoss results:")
print(f"Total Loss: {total_loss.item()}")
print(f"Reconstruction Loss: {reconstruction_loss.item()}")
print(f"KL Loss: {kl_loss.item()}")