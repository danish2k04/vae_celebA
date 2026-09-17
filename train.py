import torch
import os
from tqdm import tqdm

from dataset import train_loader, val_loader
from model import VAE
from loss import vae_loss

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ",device)

model = VAE(latent_dim=128).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

EPOCHS = 20
PATIENCE = 3

checkpoint_dir = "checkpoints"
os.makedirs("checkpoints", exist_ok=True)

best_loss = float("inf")
start_epoch = 0
patience_counter = 0

checkpoint_path = os.path.join(
    checkpoint_dir, 
    "latest_checkpoint.pth"
)

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    start_epoch = checkpoint["epoch"] + 1
    best_loss = checkpoint["best_loss"]
    patience_counter = checkpoint["patience_counter"]

    print(f"Resuming from epoch {start_epoch}")
    print(f"Best Validation Loss: {best_loss:.2f}")

train_losses = []
val_losses = []

for epoch in range(start_epoch, EPOCHS):

    model.train()
    total_epoch_loss = 0

    progress_bar = tqdm(
        train_loader, 
        desc=f"Epoch {epoch + 1}/{EPOCHS}"
    )

    for images in progress_bar:
        images = images.to(device)

        # forward pass
        reconstruction, mu, logvar = model(images)

        # Calculate Loss
        total_loss, reconstruction_loss, kl_loss = vae_loss(
            reconstruction, images, mu, logvar
        )

        optimizer.zero_grad()

        total_loss.backward()

        optimizer.step()

        total_epoch_loss += total_loss.item()

        # show current loss
        progress_bar.set_postfix(
            loss=f"{total_loss.item():.2f}"
        )
    train_loss = total_epoch_loss / len(train_loader)
    train_losses.append(train_loss)

    # Validation

    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        validation_bar = tqdm(
            val_loader,
            desc=f"Validation {epoch + 1}/{EPOCHS}"
        )

        for images in validation_bar:
            images = images.to(device)

            # forward pass
            reconstruction, mu, logvar = model(images)

            # calculate the loss
            total_loss, reconstruction_loss, kl_loss = vae_loss(
                reconstruction, images, mu, logvar
            )
            total_val_loss += total_loss.item()

            validation_bar.set_postfix(
                loss=f"{total_loss.item():.2f}"
            )

    val_loss = total_val_loss / len(val_loader) 
    val_losses.append(val_loss)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.2f} "
        f"Val Loss: {val_loss:.2f}"
    )

    if val_loss < best_loss:
        best_loss = val_loss
        patience_counter = 0

        torch.save(
            model.state_dict(),
            os.path.join(checkpoint_dir, "best_model.pth")
        )
        print("✓ Best model saved.")

    else:
        patience_counter += 1

        print(f"No improvement."
              f"Patience: {patience_counter}/{PATIENCE}")

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict":optimizer.state_dict(),
            "best_loss": best_loss,
            "patience_counter": patience_counter,
        },
        checkpoint_path
    )
    torch.save(
        {
            "train_losses": train_losses,
            "val_losses": val_losses,
        },
        os.path.join(
            checkpoint_dir, 
            "loss_history.pth"
        )
    )

    if patience_counter >= PATIENCE:
        print("Early Stopping triggered.")
        break