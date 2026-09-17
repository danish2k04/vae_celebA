import torch
import matplotlib.pyplot as plt

# load loss histroy
history = torch.load(
    "checkpoints/loss_history.pth",
    weights_only=True
)

train_losses = history["train_losses"]
val_losses = history["val_losses"]

# plot
plt.figure(figsize=(10, 6))

plt.plot(
    train_losses,
    label="Training Loss"
)

plt.plot(
    val_losses,
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("VAE Training and Validation Loss")

plt.legend()
plt.grid()

plt.show()