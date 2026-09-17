import os

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from torchvision import transforms
from PIL import Image

# 1. Transform images

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])


# 2. Custom CelebA Dataset

class CelebADataset(Dataset):

    def __init__(self, image_dir, transform=None):

        self.image_dir = image_dir
        self.transform = transform

        # Get all image filenames
        self.images = [
            filename
            for filename in os.listdir(image_dir)
            if filename.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        # Sort them so the order is predictable
        self.images.sort()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        # Get image filename
        image_name = self.images[index]

        # Build complete path
        image_path = os.path.join(self.image_dir, image_name)

        # Open image
        image = Image.open(image_path).convert("RGB")

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image



# 3. Create dataset

dataset = CelebADataset(
    image_dir="./data/img_align_celeba",
    transform=transform
)


# 4. Train / Validation split
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

# 5. Create DataLoader


train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=128,
    shuffle=False,
    num_workers=0
)


# 6. Test the dataset

if __name__ == "__main__":

    train_images = next(iter(train_loader))
    val_images = next(iter(val_loader))

    print("Total images:", len(dataset))
    print("Training images:", len(train_dataset))
    print("Validation images:", len(val_dataset))

    print("Training batch shape:", train_images.shape)
    print("Validation batch shape:", val_images.shape)

    print("Minimum pixel value:", train_images.min())
    print("Maximum pixel value:", train_images.max())