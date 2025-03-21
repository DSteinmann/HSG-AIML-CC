import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import models, transforms, datasets
import rasterio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from torch.amp import GradScaler, autocast
from torchvision.models import ResNet152_Weights
from sklearn.model_selection import train_test_split
import torch.nn.init as init
# --- Setup ---
if not os.path.exists('./outputs'):
    os.mkdir('./outputs')

# Define the paths
train_dir = './ds/images/remote_sensing/otherDatasets/sentinel_2/tif'
validation_dir = './testset/testset'  # VALIDATION data (.npy files)
model_save_path = 'resnet50_multichannel.pth'
predictions_csv_path = 'track_2.csv'

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
print(f"Using device: {DEVICE}")

BATCH_SIZE = 128
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.001

def custom_collate(batch):
    inputs = torch.stack([item[0] for item in batch], dim=0)
    labels = torch.stack([item[1] for item in batch], dim=0)
    paths = [item[2] for item in batch]
    return inputs, labels, paths
# --- Data Loading and Preprocessing ---
def load_sentinel2_image(filepath):
    """Loads a Sentinel-2 image, excluding band 10."""
    if filepath.endswith('.tif'):
        with rasterio.open(filepath) as src:
            bands = list(range(1, 10)) + list(range(11, 14))
            image = src.read(bands)
    elif filepath.endswith('.npy'):
        image = np.load(filepath)
        if image.shape[0] == 13:  # Check if band 10 is present
            image = np.concatenate((image[:9], image[10:]), axis=0)
    else:
        raise ValueError("Unsupported file type.")
    return image


def normalize_image(image):
    normalized_image = []
    for band in image:
        # Apply min-max scaling to each band
        normalized_band = (band - np.min(band)) / (np.max(band) - np.min(band))
        normalized_image.append(normalized_band)

    return np.stack(normalized_image, axis=0)


# Custom Transform for 1x1 Convolution (Keep this, but NO weight init here)
class Conv1x1(nn.Module):
    def __init__(self, in_channels=12, out_channels=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        # Weight initialization will happen in create_resnet50_model

    def forward(self, x):
        return self.conv(x)

class Sentinel2Dataset(Dataset):
    """Custom Dataset for Sentinel-2 images."""
    def __init__(self, root_dir, transform=None, file_type='.tif'):
        self.root_dir = root_dir
        self.transform = transform
        self.file_type = file_type
        self.file_paths = []
        self.labels = []

        if file_type == '.tif':
            for subdir, _, files in os.walk(root_dir):
                for file in files:
                    if file.endswith(self.file_type):
                        self.file_paths.append(os.path.join(subdir, file))
                        class_name = os.path.basename(subdir)
                        self.labels.append(class_name)

            self.classes = sorted(list(set(self.labels)))
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
            self.labels = [self.class_to_idx[label] for label in self.labels]
        #For the validation data
        elif file_type == '.npy':
            self.file_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(self.file_type)]
            self.labels = [0] * len(self.file_paths)  # Dummy labels
        else:
            raise ValueError("file_type must be '.tif' or '.npy'")



    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
      image_path = self.file_paths[idx]

      # Load based on file type.  Handle loading *before* transforms.
      if self.file_type == '.tif':
          with rasterio.open(image_path) as src:
              bands = list(range(1, 10)) + list(range(11, 14))  # Exclude band 10
              image = src.read(bands)  # (C, H, W)
      elif self.file_type == '.npy':
          image = np.load(image_path)
          if image.shape[0] == 13:  # Handle 13-band images
              image = np.concatenate((image[:9], image[10:]), axis=0)  # Combine bands
          image = image.transpose(2, 0, 1) # HWC -> CHW
      else:
          raise ValueError("Unsupported file type.")

      image = normalize_image(image)
      with torch.no_grad():  # Disable gradient tracking
          image = torch.from_numpy(image).float()  # to tenso
      label = torch.tensor(self.labels[idx], dtype=torch.long)

      if self.transform:
          image = self.transform(image)  # Now safe to use standard transforms

      return image, label, image_path #return path
# Custom Transform for 1x1 Convolution
class Conv1x1(nn.Module):
    def __init__(self, in_channels=12, out_channels=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        return self.conv(x)


# Data Augmentation
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=30),
    transforms.RandomResizedCrop(size=(64, 64), scale=(0.8, 1.0)),
])

# Validation transforms (NO augmentation, just resizing and normalization)
val_transforms = transforms.Compose([
    transforms.Resize(64),
])

# --- Create Datasets and Data Loaders ---

# Create the main training dataset (all .tif files)
full_train_dataset = Sentinel2Dataset(train_dir, transform=None, file_type='.tif')

# Split into training and validation
train_indices, val_indices = train_test_split(
    list(range(len(full_train_dataset))),
    test_size=0.2,
    random_state=42,
    stratify=full_train_dataset.labels  # Make sure to stratify by class labels!
)

train_dataset = Subset(full_train_dataset, train_indices)
val_tif_dataset = Subset(full_train_dataset, val_indices)  # Keep this for later.

# Apply transforms to the *training subset*.
train_dataset.dataset.transform = train_transforms

# Create validation dataset using .npy files, NO TRANSFORMS
validation_dataset = Sentinel2Dataset(validation_dir, transform=val_transforms, file_type='.npy')

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=32, pin_memory=True, collate_fn=custom_collate)
validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=32, pin_memory=True, collate_fn=custom_collate)

class_names = full_train_dataset.classes
print('Dataloaders OK')



# --- Model Definition (Modified ResNet50) ---
## Adjusted model creation with dropout
def create_resnet50_model(num_classes):
    model = models.resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)

    # Freeze *all* layers initially
    for param in model.parameters():
        param.requires_grad = False

    # Replace the first convolution layer to accept 12 input channels
    model.conv1 = nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # Initialize the weights of the new conv1 layer
    with torch.no_grad():
        # Load the pretrained ResNet-50
        pretrained_resnet = models.resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)
        # Copy the first 3 input channels weights from the pre-trained conv1
        model.conv1.weight.data[:, :3, :, :] = pretrained_resnet.conv1.weight.data.clone()
        # Initialize the remaining channels by duplicating the first 3 channels
        for i in range(3, 12):
            model.conv1.weight.data[:, i, :, :] = pretrained_resnet.conv1.weight.data[:, i % 3, :, :].clone()

    # Replace the final fully connected layer with dropout
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),  # Add dropout
        nn.Linear(model.fc.in_features, num_classes)
    )
    # Unfreeze the last few layers (fine-tuning)
    for param in model.fc.parameters():
        param.requires_grad = True
    for block in model.layer4:
        for param in block.parameters():
            param.requires_grad = True

    return model

model = create_resnet50_model(num_classes=len(class_names))
model.to(DEVICE) # Move model to device

# --- Loss Function, Optimizer, and Scheduler ---

criterion = nn.CrossEntropyLoss()
# *Only* optimize parameters that are set to be trainable.
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
# --- Training Loop (with Early Stopping and Mixed Precision) ---
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)
train_loss = []
train_accuracy = []
val_loss = []
val_accuracy = []
epochs_list = []

best_val_loss = float('inf')
patience = 5  # Number of epochs to wait for improvement
epochs_no_improve = 0

scaler = GradScaler()  # For mixed precision

for epoch in range(1, NUM_EPOCHS + 1):
    print(f'\nRunning epoch {epoch} of {NUM_EPOCHS}...\n')
    epochs_list.append(epoch)

    # --- Training ---
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for inputs, labels, _ in train_loader:  # Correct unpacking.
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)  # Move to GPU

        optimizer.zero_grad()

        with autocast(device_type='cuda'):  # Mixed precision
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_epoch_loss = running_loss / len(train_loader)
    train_epoch_acc = 100 * correct_train / total_train
    train_loss.append(train_epoch_loss)
    train_accuracy.append(train_epoch_acc)
    print(f'Epoch {epoch} Train Loss: {train_epoch_loss:.4f}, Train Acc: {train_epoch_acc:.2f}%')

    # --- Validation ---
    model.eval()
    running_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for inputs, labels, _ in validation_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            with autocast(device_type='cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_epoch_loss = running_loss / len(validation_loader)
    val_epoch_acc = 100 * correct_val / total_val

    # --- Scheduler Step ---
    scheduler.step(val_epoch_loss)

    # --- Save Best Model and Early Stopping ---
    if val_epoch_loss < best_val_loss:
        best_val_loss = val_epoch_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved at epoch {epoch} with validation loss: {best_val_loss:.4f}")
    else:
        epochs_no_improve += 1
        if epochs_no_improve == patience:
            print('Early stopping triggered!')
            break  # Exit the training loop

# --- Load Best Model (for Prediction) ---
best_model = create_resnet50_model(num_classes=len(class_names))  # Recreate model
best_model.load_state_dict(torch.load(model_save_path, map_location=DEVICE, weights_only=True))  # Load, move to device
best_model.to(DEVICE)
best_model.eval()  # Set to evaluation mode

# --- Prediction on Validation Set and Create CSV ---

all_predictions = []
all_filenames = []

with torch.no_grad():
    for inputs, _, image_paths in validation_loader:  # Get filenames!
        inputs = inputs.to(DEVICE)
        outputs = best_model(inputs)
        _, predicted = torch.max(outputs, 1)
        # Map numerical labels back to class names
        predicted_classes = [full_train_dataset.classes[pred.item()] for pred in predicted]
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

kaggle_competition = '8-860-1-00-coding-challenge-2025'  # Replace
kaggle_message = 'track_2'
output_csv_path = 'track_2.csv' # Output CSV
df = pd.read_csv('track_2.csv')
df['test_id'] = df['test_id'].str.replace('test_', '', regex=False)

# Save the modified DataFrame back to the CSV file
df.to_csv('track_2.csv', index=False)

try:
  #Kaggle may throw errors if it thinks the file already exist
  #this is a quick a dirty solution for it
    api.competition_submissions(kaggle_competition) #list submissions to check if the file is already there
    api.competition_submit(output_csv_path, kaggle_message, kaggle_competition) #resubmit
    print("Submission successful!")
except Exception as e:
    print(f"Submission failed: {e}")