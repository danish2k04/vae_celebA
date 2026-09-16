import os

import torch
from torch.utils.data import Dataset, DataLoader
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


# 4. Create DataLoader

loader = DataLoader(
    dataset,
    batch_size=128,
    shuffle=True,
    num_workers=0
)


# 5. Test the dataset

if __name__ == "__main__":

    images = next(iter(loader))

    print("Number of images:", len(dataset))
    print("Batch shape:", images.shape)
    print("Minimum pixel value:", images.min())
    print("Maximum pixel value:", images.max())

