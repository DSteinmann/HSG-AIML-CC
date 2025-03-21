import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import os
import rasterio
import rasterio
from kaggle.api.kaggle_api_extended import KaggleApi
from collections import Counter


# --- Setup ---
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
print(f"Using device: {DEVICE}")

BATCH_SIZE = 128
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.001

train_dir = './ds/images/remote_sensing/otherDatasets/sentinel_2/tif'
validation_dir = './testset/testset'
model_save_path = 'standalone_resnet.pth'
predictions_csv_path = 'track_1.csv'


class NpyDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_paths = []
        self.labels = []

        for file in os.listdir(root_dir):
            if file.endswith('.npy'):
                self.file_paths.append(os.path.join(root_dir, file))
                # Assuming labels are not provided, assign a dummy label for now
                self.labels.append(0)  # Replace with actual labels if available

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        data = np.load(file_path)

        # Ensure data is in the correct format: [channels, height, width]
        if data.shape[0] != 12:  # Assuming 12 channels
            data = data.transpose((2, 0, 1))  # Convert to [channels, height, width]

        data = torch.from_numpy(data).float()

        # Normalize data if necessary
        data = (data - torch.mean(data, dim=(1, 2), keepdims=True)) / (
                torch.std(data, dim=(1, 2), keepdims=True) + 1e-7)

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.transform:
            data = self.transform(data)

        return data, label, os.path.basename(file_path)
# --- Custom Dataset ---
class Sentinel2Dataset(Dataset):
    def __init__(self, root_dir, transform=None, file_type='.tif'):
        self.root_dir = root_dir
        self.transform = transform
        self.file_type = file_type
        self.file_paths = []
        self.labels = []

        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(self.file_type):
                    self.file_paths.append(os.path.join(subdir, file))
                    class_name = os.path.basename(subdir)
                    self.labels.append(class_name)

        self.classes = sorted(list(set(self.labels)))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.labels = [self.class_to_idx[label] for label in self.labels]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image_path = self.file_paths[idx]
        image = rasterio.open(image_path).read()

        # Remove the 10th band (index 9 in zero-based indexing)
        image = np.delete(image, 9, axis=0)

        # Normalize the image
        image = (image - np.mean(image, axis=(1, 2), keepdims=True)) / (
                np.std(image, axis=(1, 2), keepdims=True) + 1e-7)

        image = torch.from_numpy(image).float()
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        if self.transform:
            image = self.transform(image)
        return image, label, os.path.basename(image_path)


# --- Data Augmentation ---
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=30),
    transforms.RandomResizedCrop(size=(64, 64), scale=(0.8, 1.0)),
    transforms.Normalize([0.485] * 12, [0.229] * 12)  # Ensure this matches your data's channels
])

val_transforms = transforms.Compose([
    transforms.Resize(64),
    transforms.Normalize([0.485] * 12, [0.229] * 12)
])
# --- Create Datasets and DataLoaders ---
train_dataset = Sentinel2Dataset(train_dir, transform=train_transforms)
validation_dataset = NpyDataset(validation_dir)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)


# --- Define Standalone ResNet-like Model ---
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Ensure the residual connection has the correct number of channels
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return self.relu(x)


class StandaloneResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)

        # ResNet-like layers
        self.layer1 = ResidualBlock(64, 128)  # Downsampling here
        self.layer2 = ResidualBlock(128, 256)  # Downsampling here
        self.layer3 = ResidualBlock(256, 512)  # Downsampling here

        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(self.avgpool(x), 1)
        x = self.relu(self.fc(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


model = StandaloneResNet(num_classes=len(train_dataset.classes))
model.to(DEVICE)

# --- Training Loop ---
# Calculate class frequencies
label_counts = Counter(train_dataset.labels)
total_samples = sum(label_counts.values())
class_frequencies = {label: count / total_samples for label, count in label_counts.items()}

# Convert class frequencies to class weights
class_weights = []
for label in sorted(train_dataset.class_to_idx.keys()):
    class_weights.append(1 / class_frequencies[train_dataset.class_to_idx[label]])

class_weights = torch.tensor(class_weights).to(DEVICE)  # Move class weights to GPU
# Define the loss function with class weights
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
patience = 5  # Number of epochs to wait before stopping
early_stopping_counter = 0
best_val_loss = float('inf')
for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0
    for inputs, labels, _ in train_loader:
        #print(inputs.shape)
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss /= len(train_loader)
    accuracy = correct / total

    print(f"Epoch {epoch + 1} Train Loss: {epoch_loss:.4f}, Train Acc: {accuracy * 100:.2f}%")

    # Evaluate on validation set
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels, _ in validation_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_loss /= len(validation_loader)
    val_accuracy = correct / total

    print(f"Epoch {epoch + 1} Validation Loss: {val_loss:.4f}, Validation Acc: {val_accuracy * 100:.2f}%")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved at epoch {epoch + 1} with validation loss: {val_loss:.4f}")
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}!")
            break
# --- Save Predictions to CSV ---

# --- Prediction on Validation Set and Create CSV ---
all_predictions = []
all_filenames = []
best_model = StandaloneResNet(num_classes=len(train_dataset.classes))
best_model.load_state_dict(torch.load(model_save_path))
best_model.to(DEVICE)
best_model.eval()  # Set to evaluation mode

with torch.no_grad():
    for inputs, _, image_paths in validation_loader:  # Get filenames!
        inputs = inputs.to(DEVICE)
        outputs = best_model(inputs)
        _, predicted = torch.max(outputs, 1)
        # Map numerical labels back to class names
        predicted_classes = [train_dataset.classes[pred.item()] for pred in predicted]
        all_predictions.extend(predicted_classes)
        all_filenames.extend([os.path.basename(path) for path in image_paths])  # Correct filename

# Create the DataFrame with 'test_id' and 'label'
predictions_df = pd.DataFrame({
    'test_id': [os.path.splitext(filename)[0] for filename in all_filenames],  # Extract filename
    'label': all_predictions
})

predictions_df.to_csv(predictions_csv_path, index=False)
print(f"Validation set predictions saved to: {predictions_csv_path}")

# --- Kaggle Submission (Requires API Key) ---
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()  # Make sure your kaggle.json is set up

kaggle_competition = 'your-competition-name'  # Replace with your actual competition name
kaggle_message = 'track_1'

df = pd.read_csv('track_1.csv')
df['test_id'] = df['test_id'].str.replace('test_', '', regex=False)

# Save the modified DataFrame back to the CSV file
df.to_csv('track_1.csv', index=False)
# Upload the CSV file to Kaggle
api.competition_submit(competition=kaggle_competition, file_name=predictions_csv_path, message=kaggle_message)

print(f"Submission uploaded to Kaggle: {kaggle_competition}")