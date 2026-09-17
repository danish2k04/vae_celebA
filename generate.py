import torch
import matplotlib.pyplot as plt

from model import VAE

# device
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
print("Best Model Loaded.")

# Sample latent vectors

num_images = 16

z = torch.randn(
    num_images,
    128,
    device=device
)

# Generate Images

with torch.no_grad():
    generated_images = model.decode(z)

# Move the images to cpu
generated_images = generated_images.cpu()

# Display images
fig, axes = plt.subplots(
    4,
    4,
    figsize=(8, 8)
)

for i, ax in enumerate(axes.flat):
    ax.imshow(
        generated_images[i].permute(1, 2, 0)
    )
    ax.axis("off")

plt.tight_layout()
plt.show()