import os
import tempfile

import kornia.augmentation as K
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from torchgeo.models import ResNet18_Weights, resnet18
from torchgeo.datasets import RasterDataset
from pathlib import Path
import numpy as np
torch.manual_seed(0)
import rasterio

file_path = "./ds/images/remote_sensing/otherDatasets/sentinel_2/tif/AnnualCrop/AnnualCrop_1.tif"
try:
    with rasterio.open(file_path) as src:
        print(src.meta)
except Exception as e:
    print(f"Error opening file: {e}")


class FolderLabelDataset(RasterDataset):
    def __init__(self, root: str, transforms=None):
        super().__init__(root, transforms=transforms)
        self.files = []
        self.labels = []

        # Traverse the directory and collect file paths and labels
        for label in os.listdir(root):
            label_path = os.path.join(root, label)
            if os.path.isdir(label_path):  # Ensure it's a folder
                for file in os.listdir(label_path):
                    if file.endswith(".tif"):  # Only process .tif files
                        full_path = os.path.join(label_path, file)
                        self.files.append(full_path)
                        self.labels.append(label)

        # Map labels to integer indices
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(set(self.labels)))}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        import rasterio

        file_path = self.files[index]
        try:
            with rasterio.open(file_path) as src:
                image = src.read()  # Read all bands
        except Exception as e:
            raise RuntimeError(f"Error reading {file_path}: {e}")

        # Get the corresponding label index
        label = self.label_to_idx[self.labels[index]]

        return {"image": image, "label": label}

root_dir = os.path.abspath("./ds/images/remote_sensing/otherDatasets/sentinel_2/tif")
print(f"Normalized root directory: {root_dir}")
dataset = FolderLabelDataset(root_dir)

# Debugging: Print discovered files and labels
print(f"Number of files: {len(dataset)}")
print("Sample files:")
for i in range(min(5, len(dataset))):  # Print up to 5 sample files
    print(dataset.files[i])
print("Labels:", dataset.label_to_idx)

for i in torch.randint(len(dataset), (10,)):
    sample = dataset[i]
    dataset.plot(sample)

indices = np.arange(len(dataset))
np.random.shuffle(indices)
train_indices, val_indices, test_indices = np.split(indices, [int(0.7 * len(indices)), int(0.9 * len(indices))])

train_dataset = torch.utils.data.Subset(dataset, train_indices)
val_dataset = torch.utils.data.Subset(dataset, val_indices)
test_dataset = torch.utils.data.Subset(dataset, test_indices)

batch_size = 10
epochs = 100

train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False)

preprocess = K.Normalize(0, 10000)
augment = K.ImageSequential(K.RandomHorizontalFlip(), K.RandomVerticalFlip())

model = resnet18(ResNet18_Weights.SENTINEL2_ALL_MOCO)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-2)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

def train(dataloader):
    model.train()
    total_loss = 0
    for batch in dataloader:
        x = batch['image'].to(device)
        y = batch['label'].to(device)

        # Forward pass
        y_hat = model(x)
        loss = loss_fn(y_hat, y)
        total_loss += loss.item()

        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f'Loss: {total_loss:.2f}')

def evaluate(dataloader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch in dataloader:
            x = batch['image'].to(device)
            y = batch['label'].to(device)

            # Forward pass
            y_hat = model(x)
            correct += (y_hat.argmax(1) == y).type(torch.float).sum().item()

    correct /= len(dataloader.dataset)
    print(f'Accuracy: {correct:.0%}')

for epoch in range(epochs):
    print(f'Epoch: {epoch}')
    train(train_dataloader)
    evaluate(val_dataloader)

evaluate(test_dataloader)