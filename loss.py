import torch
import torch.nn.functional as F

def vae_loss(reconstruction, original, mu, logvar):

    # Reconstruction loss
    reconstruction_loss = F.binary_cross_entropy(
        reconstruction, original, reduction="sum"
    )

    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Total VAE loss
    total_loss = reconstruction_loss + kl_loss

    return total_loss, reconstruction_loss, kl_loss
