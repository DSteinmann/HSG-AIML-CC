import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import rasterio
import zipfile
import glob

# Define dataset paths
train_zip_path = "C:/Users/xadri/OneDrive/Dokumente/Master/2Semester/ML/X_MS.zip" #MS dataset
test_zip_path = "C:/Users/xadri/OneDrive/Dokumente/Master/2Semester/ML/X.zip"     #kaggle testset
train_dir = "C:/Users/xadri/OneDrive/Dokumente/Master/2Semester/ML/EuroSAT_train"
test_dir = "C:/Users/xadri/OneDrive/Dokumente/Master/2Semester/ML/EuroSAT_test"

# Extract datasets
with zipfile.ZipFile(train_zip_path, "r") as zip_ref:
    zip_ref.extractall(train_dir)
with zipfile.ZipFile(test_zip_path, "r") as zip_ref:
    zip_ref.extractall(test_dir)

# Define Dataset Class
class EuroSATDataset(Dataset):
    def __init__(self, image_paths, labels=None, transform=None, is_npy=False):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.is_npy = is_npy
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        if self.is_npy:
            img = np.load(self.image_paths[idx]).astype(np.float32)
        else:
            with rasterio.open(self.image_paths[idx]) as src:
                img = src.read().astype(np.float32)  # Read all bands
        
        img = torch.tensor(img, dtype=torch.float32)
        if self.transform:
            img = self.transform(img)
        if self.labels:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return img, label
        return img

# Get image paths and labels for training
train_image_paths = glob.glob(os.path.join(train_dir, "*/*.tif"))
train_labels = [os.path.basename(os.path.dirname(p)) for p in train_image_paths]
label_dict = {label: idx for idx, label in enumerate(sorted(set(train_labels)))}
train_labels = [label_dict[label] for label in train_labels]

# Data Augmentation & Normalization
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.Normalize(mean=[0.5]*12, std=[0.5]*12)  # Adjust as needed
])

# Load Training Data
train_dataset = EuroSATDataset(image_paths=train_image_paths, labels=train_labels, transform=transform, is_npy=False)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Modify ResNet for 12-band Input
class ResNet12Band(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet12Band, self).__init__()
        self.resnet = models.resnet18(weights=None)  # No pretrained weights
        self.resnet.conv1 = nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        return self.resnet(x)

# Model, Loss, Optimizer
model = ResNet12Band(num_classes=10).cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

print("Training complete!")

# Save model
torch.save(model.state_dict(), "eurosat_model.pth")

# Get test image paths
test_image_paths = glob.glob(os.path.join(test_dir, "*.npy"))

def predict(model, image_path, is_npy=True):
    model.eval()
    with torch.no_grad():
        if is_npy:
            img = np.load(image_path).astype(np.float32)
        else:
            with rasterio.open(image_path) as src:
                img = src.read().astype(np.float32)
        
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).cuda()  # Add batch dimension
        output = model(img)
        predicted_class = torch.argmax(output, dim=1).item()
    return predicted_class

# Example usage
model.load_state_dict(torch.load("eurosat_model.pth"))  # Load trained model
for test_image_path in test_image_paths:
    predicted_class = predict(model, test_image_path, is_npy=True)
    print(f"Predicted class for {test_image_path}: {predicted_class}")
