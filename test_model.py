import torch 

from dataset import loader
from model import VAE


# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# create a model
model = VAE(latent_dim=128).to(device)
print("\nModel created successfuly.")

# get one batch
images = next(iter(loader)).to(device)
print(f"\nInput shape: {images.shape}")

# Forward pass
reconstruction, mu, logvar = model(images)

# print the output
print("Reconstruction shape: ", reconstruction.shape)
print("Mu shape: ",mu.shape)
print("Logvar shape: ",logvar.shape)