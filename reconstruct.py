import torch
import matplotlib.pyplot as plt

from dataset import val_loader
from model import VAE

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = VAE(latent_dim=128).to(device)

model.load_state_dict(
    torch.load(
        "checkpoints/best_model.pth",
        map_location=device,
        weights_only=True
    )
)

model.eval()
print("Best model loaded.")

# Get Validation images

images = next(iter(val_loader))
images= images.to(device)

# Reconstruction
with torch.no_grad():
    reconstruction, mu, logvar = model(images)


# Move images to CPU
images = images.cpu()
reconstruction = reconstruction.cpu()

# Display results
fig, axes = plt.subplots(
    2, 
    8,
    figsize=(16, 4)
)

for i in range(8):
    # Original
    axes[0, i].imshow(
        images[i].permute(1, 2, 0)
    )

    axes[0, i].axis("off")

    # Reconstruction
    axes[1, i].imshow(
        reconstruction[i].permute(1, 2, 0)
    )
    axes[1, i].axis("off")

axes[0, 0].set_ylabel(
    "Original",
    fontsize=12
)
axes[1, 0].set_ylabel(
    "Reconstruction",
    fontsize=12
)

plt.tight_layout()
plt.show()