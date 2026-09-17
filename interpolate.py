import torch
import matplotlib.pyplot as plt

from dataset import val_loader
from model import VAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VAE(latent_dim=128).to(device)

model.load_state_dict(
    torch.load(
        "checkpoints/best_model.pth",
        map_location=device,
        weights_only=True
    )
)
model.eval()

# Get Real Faces
images = next(iter(val_loader))
images = images.to(device)

image_a = images[0].unsqueeze(0)
image_b = images[1].unsqueeze(0)

# Encode both faces
z_a, _, _ = model.encode(image_a)
z_b, _, _ = model.encode(image_b)


# Create the Interpolation
steps = 8
alphas = torch.linspace(
    0,
    1,
    steps,
    device=device
)

interpolated_z = torch.cat(
    [
        (1 - alpha) * z_a + alpha * z_b for alpha in alphas
    ],
    dim=0
)

# Decode 

with torch.no_grad():
    interpolated_images = model.decode(
        interpolated_z
    )

interpolated_images = interpolated_images.cpu()

# Diplay the interpolation
fig, axes = plt.subplots(
    1,
    steps,
    figsize=(16,3)
)

for i, ax in enumerate(axes):
    ax.imshow(
        interpolated_images[i].permute(1, 2, 0)
    )
    ax.axis("off")

plt.tight_layout()
plt.show()