# -*- coding: utf-8 -*-
"""
Runnable script combining CycleGAN Training and Sentinel-2 Classifier Training,
integrating the GAN generator for online data augmentation during classifier training.

Dependencies: torch, torchvision, rasterio, numpy, pandas, scikit-learn, tqdm
"""

# --- Suppress warnings (optional) ---
import warnings
warnings.filterwarnings("ignore")

# --- Standard Imports ---
import os
import random
import time
import traceback
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional, Callable

# --- ML/Data Science Imports ---
import numpy as np
import pandas as pd
import rasterio
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- PyTorch Imports ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import models, transforms
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torchvision.models import ResNet50_Weights, ConvNeXt_Base_Weights # Note: These weights aren't used by default in the custom models below
from torch.utils.checkpoint import checkpoint

# --- Environment Setup ---
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True' # May help with CUDA memory fragmentation

# --- GPU/Device Setup ---
torch.cuda.empty_cache()  # Clear cache at start
torch.backends.cudnn.benchmark = True  # Optimizes conv algo selection if input sizes don't vary much

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
print(f"Using device: {DEVICE}")

# ==================================
# === Checkpoint Backport (if needed) ===
# ==================================
# (Keep the checkpoint_sequential function as defined in the previous version)
def checkpoint_sequential(module, x):
    """
    Applies checkpointing to each module in a sequential block for memory saving.
    """
    def run_function(start, end, current_module, current_input):
        # This function will be checkpointed
        for i in range(start, end):
             current_input = current_module[i](current_input)
        return current_input

    # Get the sequential module's children
    if isinstance(module, nn.Sequential):
        sub_modules = list(module.children())
    else: # Handle single module case if needed, though intended for Sequential
        return checkpoint(module, x)

    segments = len(sub_modules)
    if segments <= 1: # No need to checkpoint if only 0 or 1 sub-module
        return module(x)

    # Process segment by segment
    segment_size = 1 # Checkpoint each sub-module individually
    current_input = x
    for start in range(0, segments, segment_size):
        end = min(start + segment_size, segments)
        # Pass the relevant sub_modules for the current segment
        current_input = checkpoint(run_function, start, end, nn.ModuleList(sub_modules), current_input)

    return current_input

# ==================================
# === CBAM Attention Mechanism ===
# ==================================
# (Keep the CBAM class as defined in the previous version)
class CBAM(nn.Module):
    def __init__(self, channels, reduction=8):  # Reduced reduction ratio
        super().__init__()
        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # Use Conv2d with kernel size 1 for linear projection simulation
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid() # Renamed for clarity

        # Spatial Attention
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False) # Kernel size 7 needs padding 3
        self.sigmoid_spatial = nn.Sigmoid() # Renamed for clarity

    def forward(self, x):
        # Channel Attention Path
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_att = self.sigmoid_channel(avg_out + max_out)
        x_channel_att = x * channel_att # Apply channel attention

        # Spatial Attention Path
        # Reduce along channel dimension after channel attention
        avg_out_spatial = torch.mean(x_channel_att, dim=1, keepdim=True)
        max_out_spatial, _ = torch.max(x_channel_att, dim=1, keepdim=True)
        spatial_cat = torch.cat([avg_out_spatial, max_out_spatial], dim=1) # Concat along channel dim (dim=1)
        spatial_att = self.sigmoid_spatial(self.conv_spatial(spatial_cat))

        # Apply spatial attention to the channel-attended feature map
        x_out = x_channel_att * spatial_att
        return x_out

# ==================================
# === CycleGAN Generator ===
# ==================================
# (Keep the ResNetBlock and Generator classes as defined in the previous version)
class ResNetBlock(nn.Module):
    def __init__(self, channels):
        super(ResNetBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(channels), # Use InstanceNorm for GANs typically
            nn.ReLU(inplace=True), # Inplace ReLU
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x) # Residual connection

class Generator(nn.Module):
    def __init__(self, in_channels=12, out_channels=12, num_res_blocks=6):
        super(Generator, self).__init__()

        # Initial convolution block
        layers = [
            nn.ReflectionPad2d(3), # Padding before conv
            nn.Conv2d(in_channels, 64, kernel_size=7, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]

        # Downsampling
        curr_dim = 64
        for _ in range(2): # Two downsampling layers
            layers += [
                nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(curr_dim * 2),
                nn.ReLU(inplace=True)
            ]
            curr_dim *= 2 # 64 -> 128 -> 256

        # Residual blocks
        for _ in range(num_res_blocks):
            layers += [ResNetBlock(curr_dim)] # Should be 256 channels here

        # Upsampling
        for _ in range(2): # Two upsampling layers
            layers += [
                nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                nn.InstanceNorm2d(curr_dim // 2),
                nn.ReLU(inplace=True)
            ]
            curr_dim //= 2 # 256 -> 128 -> 64

        # Output layer
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(curr_dim, out_channels, kernel_size=7), # Should be 64 channels here
            nn.Tanh() # Output normalized to [-1, 1] <--- IMPORTANT: Mismatch with classifier normalization!
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # Apply checkpointing to the sequential model
        try:
             return checkpoint_sequential(self.model, x)
        except Exception as e:
            print(f"Warning: Checkpointing failed in Generator: {e}. Running model sequentially.")
            return self.model(x)

# ==================================
# === CycleGAN Discriminator ===
# ==================================
# (Keep the Discriminator class as defined in the previous version)
class Discriminator(nn.Module):
    def __init__(self, in_channels=12):
        super().__init__()
        # PatchGAN architecture
        layers = [
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1), # Output: N x 64 x H/2 x W/2
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False), # Output: N x 128 x H/4 x W/4
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False), # Output: N x 256 x H/8 x W/8
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, bias=False), # Output: N x 512 x H/8-ish x W/8-ish (depends on padding calculation)
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # Output layer: 1 channel prediction map (PatchGAN)
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1) # Output: N x 1 x H/8-ish x W/8-ish
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x) # Output is a map of predictions, not a single value

# ==================================
# === Progressive Resizing (CycleGAN) ===
# ==================================
# (Keep progressive_resize_gan function - Note: Not actively used in GAN loop below)
# Global variable for transform, might be better encapsulated
transform_train_gan = None # Initialize

def progressive_resize_gan(epoch, initial_size=64, max_size=192, total_epochs_gan=50):
    """ Gradual size increase for GAN training """
    growth_interval = 10 # Increase size every 10 epochs
    num_steps = (max_size - initial_size) // 8 # Number of size increments (step of 8)
    epochs_per_step = max(1, total_epochs_gan // (num_steps + 1)) # Distribute epochs across sizes

    current_step = min(epoch // epochs_per_step, num_steps)
    new_size = initial_size + current_step * 8
    new_size = min(new_size, max_size) # Cap at max size

    global transform_train_gan
    transform_train_gan = transforms.Compose([
        transforms.Resize((new_size, new_size), antialias=True),
    ])
    return new_size

# ==================================
# === Configuration Dictionary ===
# ==================================
CONFIG = {
    "model": {
        "load_path": Path('./outputs/sentinel_classifier_best.pth'), # Save path for the classifier
        "num_classes": None,
        "class_names": None,
    },
    "data": {
        "train_dir": Path('./ds/images/remote_sensing/otherDatasets/sentinel_2/tif'),
        "validation_dir": Path('./testset/testset'),
        "image_size": 128, # Target size for the classifier
        "batch_size": 32,
        "num_workers": min(os.cpu_count(), 4),
        "train_ratio": 0.9,
    },
    "train": {
        "seed": 1337,
        "lr_stage1": 1e-3,
        "lr_stage2": 1e-4,
        "stage1_epochs": 30, # Consider reducing if GAN aug is slow
        "stage2_epochs": 30, # Consider reducing if GAN aug is slow
        "warmup_epochs": 3,
        "initial_warmup_lr": 1e-6,
        "weight_decay": 3.7553e-05,
        "optimizer_name": 'AdamW',
        "patience": 10,
        "gradient_accumulation_steps": 2,
        # --- GAN Augmentation Config ---
        "use_gan_augmentation": True, # Set to True to enable GAN augmentation
        "gan_augmentation_prob": 0.3, # Probability of applying GAN augmentation to a batch
        "gan_epochs": 10 # Number of epochs to train the GAN (adjust as needed)

    },
    "device": DEVICE.type,
    "amp_enabled": USE_CUDA,
    "prediction": {
        "predictions_csv_path": Path('./outputs/predictions.csv'),
        "kaggle_competition": 'aicrowd-geospatial-challenge',
        "kaggle_message": 'Submission message',
        "submit_to_kaggle": False,
    }
}

# ==================================
# === Seed Setting ===
# ==================================
# (Keep set_seed function as defined in the previous version)
def set_seed(seed: int):
    """Sets random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Keep benchmark=True for speed, but set deterministic=False
        # For full determinism (slower):
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    print(f"Seed set to {seed}")

set_seed(CONFIG["train"]["seed"])

# ==================================
# === Data Loading & Preprocessing ===
# ==================================
# (Keep load_sentinel2_image and normalize_image_per_image functions)
def load_sentinel2_image(filepath: str) -> Optional[np.ndarray]:
    """Loads a Sentinel-2 image (TIF or NPY), returns NumPy CHW (12 bands)."""
    try:
        if filepath.lower().endswith(('.tif', '.tiff')):
            with rasterio.open(filepath) as src:
                if src.count < 13:
                    # print(f"Warning: Expected >=13 bands, got {src.count} in {filepath}. Skipping B10 if possible.")
                    if src.count == 12: # Assume bands 1-9, 11, 12, 13 are present
                         bands_to_read = list(range(1, 13)) # Read all 12 bands
                    else:
                         # Cannot safely assume band structure if not 12 or >=13
                         print(f"Error: Unexpected band count {src.count} in {filepath}. Cannot determine bands.")
                         return None
                else:
                    # Bands 1-9, 11, 11A, 12 (Indices 0-8, 10, 11, 12 in 0-based)
                    # Exclude B10 (index 9)
                    bands_to_read = list(range(1, 10)) + list(range(11, 14)) # Sentinel-2 Bands: 1-9, 11, 11A, 12

                # Check if bands_to_read are valid for the file
                valid_bands = [b for b in bands_to_read if b <= src.count]
                if len(valid_bands) != 12:
                    # print(f"Warning: Could not read all desired 12 bands from {filepath}. Got {len(valid_bands)}. Reading available bands.")
                     # Attempt to read the first 12 available bands if standard selection failed
                     if src.count >= 12:
                         valid_bands = list(range(1, 13))
                     else:
                         print(f"Error: Not enough bands ({src.count}) to form a 12-band image in {filepath}.")
                         return None

                image = src.read(valid_bands) # Shape: (12, H, W)


        elif filepath.lower().endswith('.npy'):
            image = np.load(filepath)
            # Assume NPY is already in CHW format (12 or 13 channels)
            if image.shape[0] == 13:
                 # Assume standard 13 bands and exclude B10 (index 9)
                 indices = list(range(9)) + list(range(10, 13))
                 image = image[indices, :, :] # Select 12 bands
            elif image.shape[0] != 12:
                 print(f"Error: Unexpected channel count {image.shape[0]} in .npy file {filepath}. Expected 12 or 13.")
                 return None
            # If shape[0] is 12, assume it's already the correct 12 bands

        else:
            print(f"Error: Unsupported file type: {filepath}")
            return None

        # Ensure float32 type
        return image.astype(np.float32)

    except rasterio.RasterioIOError as e:
        print(f"Rasterio I/O Error loading {filepath}: {e}")
        return None
    except Exception as e:
        print(f"Generic Error loading {filepath}: {e}")
        traceback.print_exc()
        return None


def normalize_image_per_image(image_np: np.ndarray) -> Optional[np.ndarray]:
    """Normalizes 12-channel NumPy image (C, H, W) using its own stats."""
    if image_np is None: return None
    if image_np.ndim != 3 or image_np.shape[0] != 12:
        print(f"Error: Invalid shape for per-image normalization: {image_np.shape}. Expected (12, H, W).")
        return None

    # Calculate per-channel mean and std
    # Ensure calculation happens on valid data (avoid NaN/inf issues)
    if np.isnan(image_np).any() or np.isinf(image_np).any():
        print("Warning: NaN or Inf values found in image before normalization. Attempting to clean.")
        image_np = np.nan_to_num(image_np, nan=0.0, posinf=1.0, neginf=0.0) # Replace NaNs/Infs

    mean = np.mean(image_np, axis=(1, 2), keepdims=True, dtype=np.float32)
    std = np.std(image_np, axis=(1, 2), keepdims=True, dtype=np.float32)

    # Avoid division by zero or near-zero std dev
    std[std < 1e-7] = 1e-7

    normalized_image = (image_np - mean) / std
    return normalized_image

# ==================================
# === Dataset Classes ===
# ==================================
# (Keep Sentinel2Dataset and NpyPredictionDataset classes)
class Sentinel2Dataset(Dataset):
    """Custom Dataset for Sentinel-2 TIF images (Training/Validation Split)."""
    def __init__(self, paths_labels: List[Tuple[str, int]], transform: Optional[Callable] = None):
        self.paths_labels = paths_labels
        self.transform = transform
        print(f"Initialized Sentinel2Dataset with {len(paths_labels)} samples.")

    def __len__(self):
        return len(self.paths_labels)

    def __getitem__(self, idx):
        image_path, label = self.paths_labels[idx]
        try:
            image_np = load_sentinel2_image(image_path) # NumPy (C, H, W) float32 or None
            if image_np is None: # Handle loading failure
                 # print(f"Warning: Skipping item {idx} due to image loading error: {image_path}")
                 return None # Signal error to collate_fn

            image_np_normalized = normalize_image_per_image(image_np) # Apply per-image normalization
            if image_np_normalized is None: # Handle normalization failure
                 # print(f"Warning: Skipping item {idx} due to image normalization error: {image_path}")
                 return None

            # Convert to tensor AFTER normalization
            image_tensor = torch.from_numpy(image_np_normalized.copy()).float()
            label_tensor = torch.tensor(label, dtype=torch.long)

            # Apply other transforms (like resizing, augmentation) if provided
            if self.transform:
                image_tensor = self.transform(image_tensor)

            return image_tensor, label_tensor, image_path

        except Exception as e:
            print(f"Error in __getitem__ for index {idx}, path {image_path}:")
            traceback.print_exc()
            return None # Signal error to collate_fn


class NpyPredictionDataset(Dataset):
    """Dataset for loading NPY files for final prediction."""
    def __init__(self, root_dir: str, transform: Optional[Callable] = None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        try:
            self.file_paths = sorted([str(p) for p in self.root_dir.glob('*.npy')])
            if not self.file_paths:
                raise FileNotFoundError(f"No .npy files found in {self.root_dir}")
            print(f"Found {len(self.file_paths)} .npy files for prediction in {self.root_dir}.")
        except Exception as e:
             print(f"Error initializing NpyPredictionDataset: {e}")
             self.file_paths = [] # Ensure it's an empty list on error

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        if idx >= len(self.file_paths): # Safety check
            print(f"Error: Index {idx} out of bounds for NpyPredictionDataset.")
            return None

        image_path = self.file_paths[idx]
        try:
            image_np = load_sentinel2_image(image_path) # NumPy (C, H, W) float32 or None
            if image_np is None:
                 # print(f"Warning: Skipping prediction item {idx} due to image loading error: {image_path}")
                 return None

            image_np_normalized = normalize_image_per_image(image_np) # Normalize
            if image_np_normalized is None:
                 # print(f"Warning: Skipping prediction item {idx} due to normalization error: {image_path}")
                 return None

            image_tensor = torch.from_numpy(image_np_normalized.copy()).float() # Convert to tensor

            if self.transform:
                image_tensor = self.transform(image_tensor) # Apply transforms (e.g., resizing)

            # Return image tensor, dummy label (0), and image path
            return image_tensor, torch.tensor(0, dtype=torch.long), image_path

        except Exception as e:
            print(f"Error in NpyPredictionDataset __getitem__ for index {idx}, path {image_path}:")
            traceback.print_exc()
            return None

# ==================================
# === Data Transforms ===
# ==================================
# (Keep AddGaussianNoise, RandomChannelScale, train_transforms, val_transforms)
class AddGaussianNoise(object):
    """Adds Gaussian noise to a tensor."""
    def __init__(self, mean=0., std=0.1):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'

class RandomChannelScale(object):
    """Applies random scaling to each channel independently."""
    def __init__(self, scale_range=(0.9, 1.1)):
        self.scale_range = scale_range

    def __call__(self, tensor):
        if tensor.dim() != 3: # Expect CHW
            print(f"Warning: RandomChannelScale expects 3D tensor (CHW), got {tensor.dim()}D. Skipping.")
            return tensor
        scale_factors = torch.empty(tensor.shape[0]).uniform_(self.scale_range[0], self.scale_range[1])
        return tensor * scale_factors.view(-1, 1, 1) # Reshape for broadcasting (C, 1, 1)

    def __repr__(self):
        return self.__class__.__name__ + f'(scale_range={self.scale_range})'

train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=20),
    AddGaussianNoise(mean=0., std=0.05),
    RandomChannelScale(scale_range=(0.9, 1.1)),
    transforms.Resize((CONFIG["data"]["image_size"], CONFIG["data"]["image_size"]), antialias=True),
])

val_transforms = transforms.Compose([
    transforms.Resize((CONFIG["data"]["image_size"], CONFIG["data"]["image_size"]), antialias=True),
])

# ==================================
# === Dataset & DataLoader Creation ===
# ==================================
# (Keep the dataset scanning, splitting, and DataLoader creation logic)
print("Scanning training directory and creating dataset splits...")
full_dataset_paths_labels = []
class_to_idx_map = {}
class_names_list = [] # Use a list to preserve order
idx_counter = 0

train_dir_path = CONFIG["data"]["train_dir"]
if not train_dir_path.is_dir():
     raise FileNotFoundError(f"Training directory not found: {train_dir_path}")

# Scan training directory for class folders
for class_name in sorted(os.listdir(train_dir_path)):
     class_dir = train_dir_path / class_name
     if class_dir.is_dir() and not class_name.startswith('.'):
         if class_name not in class_to_idx_map:
             class_to_idx_map[class_name] = idx_counter
             class_names_list.append(class_name)
             idx_counter += 1
         class_idx = class_to_idx_map[class_name]
         image_count = 0
         # Find .tif files within the class directory
         for filename in os.listdir(class_dir):
             if filename.lower().endswith(('.tif', '.tiff')):
                 full_dataset_paths_labels.append((str(class_dir / filename), class_idx))
                 image_count += 1
         # print(f"  Found {image_count} images in class '{class_name}' (Index: {class_idx})")


num_classes = len(class_names_list)
if num_classes == 0:
    raise FileNotFoundError(f"No valid class folders containing .tif files found in {train_dir_path}")

CONFIG["model"]["num_classes"] = num_classes
CONFIG["model"]["class_names"] = class_names_list
print(f"Found {len(full_dataset_paths_labels)} total training images in {num_classes} classes: {class_names_list}")

# Stratified Split
try:
    train_info, val_info = train_test_split(
        full_dataset_paths_labels,
        train_size=CONFIG["data"]["train_ratio"],
        random_state=CONFIG["train"]["seed"],
        stratify=[label for _, label in full_dataset_paths_labels] # Stratify based on labels
    )
    print(f"Split dataset: {len(train_info)} training samples, {len(val_info)} validation samples.")
except ValueError as e:
     print(f"Error during train/test split (possibly too few samples per class for stratification): {e}")
     # Fallback: random split if stratification fails
     train_info, val_info = train_test_split(
        full_dataset_paths_labels,
        train_size=CONFIG["data"]["train_ratio"],
        random_state=CONFIG["train"]["seed"]
     )
     print(f"Using random split instead: {len(train_info)} training samples, {len(val_info)} validation samples.")


# Create Dataset objects
train_dataset = Sentinel2Dataset(train_info, transform=train_transforms)
val_tif_dataset = Sentinel2Dataset(val_info, transform=val_transforms) # Validation uses TIFs from split

# Create Prediction dataset (using .npy files from a separate directory)
validation_dir_path = CONFIG["data"]["validation_dir"]
if not validation_dir_path.is_dir():
     print(f"Warning: Final prediction directory not found: {validation_dir_path}. Evaluation step will fail.")
     final_validation_dataset = None # Mark as None if dir doesn't exist
else:
     final_validation_dataset = NpyPredictionDataset(str(validation_dir_path), transform=val_transforms)

# --- Create DataLoaders ---
# (Keep collate_fn)
def collate_fn(batch):
    """ Filters out None samples potentially returned by Dataset __getitem__ on error. """
    batch = [item for item in batch if item is not None]
    if not batch:
        return None, None, None
    try:
        images = torch.stack([item[0] for item in batch])
        labels = torch.stack([item[1] for item in batch])
        paths = [item[2] for item in batch]
        return images, labels, paths
    except Exception as e:
        print(f"Error in collate_fn during stacking: {e}. Skipping batch.")
        return None, None, None

g = torch.Generator()
g.manual_seed(CONFIG["train"]["seed"])

train_loader = DataLoader(
    train_dataset, batch_size=CONFIG["data"]["batch_size"], shuffle=True,
    num_workers=CONFIG["data"]["num_workers"], generator=g, pin_memory=True,
    prefetch_factor=2 if CONFIG["data"]["num_workers"] > 0 else None,
    persistent_workers=True if CONFIG["data"]["num_workers"] > 0 else False,
    collate_fn=collate_fn, drop_last=True
)

val_loader_split = DataLoader(
    val_tif_dataset, batch_size=CONFIG["data"]["batch_size"] * 2, shuffle=False,
    num_workers=CONFIG["data"]["num_workers"], pin_memory=True,
    prefetch_factor=2 if CONFIG["data"]["num_workers"] > 0 else None,
    persistent_workers=True if CONFIG["data"]["num_workers"] > 0 else False,
    collate_fn=collate_fn
)

if final_validation_dataset:
    final_pred_loader = DataLoader(
        final_validation_dataset, batch_size=CONFIG["data"]["batch_size"] * 2, shuffle=False,
        num_workers=CONFIG["data"]["num_workers"], pin_memory=True,
        prefetch_factor=2 if CONFIG["data"]["num_workers"] > 0 else None,
        persistent_workers=True if CONFIG["data"]["num_workers"] > 0 else False,
        collate_fn=collate_fn
    )
    print("DataLoaders for training, validation split, and final prediction created.")
else:
    final_pred_loader = None
    print("DataLoaders for training and validation split created. Final prediction loader skipped (dataset not found).")


# ==================================
# === CycleGAN Initialization & Training ===
# ==================================
# This section trains the GAN models FIRST. generator_XY will be used later.
print("\n--- Initializing CycleGAN Models ---")
generator_XY = Generator(in_channels=12, out_channels=12).to(DEVICE)
generator_YX = Generator(in_channels=12, out_channels=12).to(DEVICE)
discriminator_X = Discriminator(in_channels=12).to(DEVICE)
discriminator_Y = Discriminator(in_channels=12).to(DEVICE)

# Losses & Optimizers
cycle_loss_fn = nn.L1Loss()
identity_loss_fn = nn.L1Loss()
gan_loss_fn = nn.MSELoss() # LSGAN

optimizer_G = torch.optim.Adam(
    list(generator_XY.parameters()) + list(generator_YX.parameters()),
    lr=2e-4, betas=(0.5, 0.999)
)
optimizer_D = torch.optim.Adam(
    list(discriminator_X.parameters()) + list(discriminator_Y.parameters()),
    lr=2e-4, betas=(0.5, 0.999)
)

lambda_cycle = 10.0
lambda_identity = 0.5
num_epochs_gan = CONFIG["train"]["gan_epochs"] # Use configured number of epochs

print(f"--- Starting CycleGAN Training ({num_epochs_gan} epochs) ---")
# (Keep the GAN training loop as before)
# GAN Training Loop
for epoch_gan in range(num_epochs_gan):
    gen_losses = []
    disc_losses = []

    pbar_gan = tqdm(train_loader, desc=f"GAN Epoch {epoch_gan+1}/{num_epochs_gan}")
    # Need a separate iterator for real_Y to avoid consuming train_loader too quickly if needed
    real_y_iterator = iter(train_loader)

    for i, batch_data in enumerate(pbar_gan):
        if batch_data is None or batch_data[0] is None: continue
        real_X, _, _ = batch_data

        # Get real_Y using the separate iterator
        try:
             batch_Y_data = next(real_y_iterator)
             # If iterator exhausted, reset it (common in GAN training)
             if batch_Y_data is None or batch_Y_data[0] is None:
                 real_y_iterator = iter(train_loader) # Reset iterator
                 batch_Y_data = next(real_y_iterator)
                 if batch_Y_data is None or batch_Y_data[0] is None: continue # Skip if still invalid
             real_Y, _, _ = batch_Y_data
        except StopIteration:
             # print("GAN loop: Resetting real_Y iterator.")
             real_y_iterator = iter(train_loader) # Reset iterator
             try:
                 batch_Y_data = next(real_y_iterator)
                 if batch_Y_data is None or batch_Y_data[0] is None: continue
                 real_Y, _, _ = batch_Y_data
             except StopIteration:
                 print("Error: Could not get data from reset iterator for real_Y. Breaking GAN inner loop.")
                 break # Should not happen

        # Minimal batch size check
        if real_X.shape[0] < 2 or real_Y.shape[0] < 2: continue # Need at least 2 for batch norm stability maybe?
        # Use smaller batch size if they differ
        min_batch_size = min(real_X.shape[0], real_Y.shape[0])
        real_X = real_X[:min_batch_size].to(DEVICE)
        real_Y = real_Y[:min_batch_size].to(DEVICE)


        # --- Train Generators ---
        optimizer_G.zero_grad()

        # Identity loss
        if lambda_identity > 0:
            identity_Y = generator_XY(real_Y)
            loss_identity_Y = identity_loss_fn(identity_Y, real_Y)
            identity_X = generator_YX(real_X)
            loss_identity_X = identity_loss_fn(identity_X, real_X)
            total_identity_loss = (loss_identity_X + loss_identity_Y) * lambda_identity * lambda_cycle
        else:
            total_identity_loss = 0.0

        # GAN loss
        fake_Y = generator_XY(real_X)
        pred_fake_Y = discriminator_Y(fake_Y)
        target_tensor_real = torch.ones_like(pred_fake_Y, requires_grad=False).to(DEVICE)
        loss_gan_XY = gan_loss_fn(pred_fake_Y, target_tensor_real)

        fake_X = generator_YX(real_Y)
        pred_fake_X = discriminator_X(fake_X)
        target_tensor_real_x = torch.ones_like(pred_fake_X, requires_grad=False).to(DEVICE) # Potentially different size
        loss_gan_YX = gan_loss_fn(pred_fake_X, target_tensor_real_x)

        # Cycle consistency loss
        reconstructed_X = generator_YX(fake_Y)
        loss_cycle_X = cycle_loss_fn(reconstructed_X, real_X)

        reconstructed_Y = generator_XY(fake_X)
        loss_cycle_Y = cycle_loss_fn(reconstructed_Y, real_Y)
        total_cycle_loss = (loss_cycle_X + loss_cycle_Y) * lambda_cycle

        loss_G = loss_gan_XY + loss_gan_YX + total_cycle_loss + total_identity_loss
        gen_losses.append(loss_G.item())
        loss_G.backward()
        optimizer_G.step()

        # --- Train Discriminators ---
        optimizer_D.zero_grad()

        # Discriminator X
        pred_real_X = discriminator_X(real_X)
        loss_D_real_X = gan_loss_fn(pred_real_X, target_tensor_real_x) # Use x-sized target
        pred_fake_X = discriminator_X(fake_X.detach())
        target_tensor_fake_x = torch.zeros_like(pred_fake_X, requires_grad=False).to(DEVICE)
        loss_D_fake_X = gan_loss_fn(pred_fake_X, target_tensor_fake_x)
        loss_D_X = (loss_D_real_X + loss_D_fake_X) * 0.5

        # Discriminator Y
        pred_real_Y = discriminator_Y(real_Y)
        loss_D_real_Y = gan_loss_fn(pred_real_Y, target_tensor_real) # Use y-sized target (same as original fake_Y target)
        pred_fake_Y = discriminator_Y(fake_Y.detach())
        target_tensor_fake_y = torch.zeros_like(pred_fake_Y, requires_grad=False).to(DEVICE)
        loss_D_fake_Y = gan_loss_fn(pred_fake_Y, target_tensor_fake_y)
        loss_D_Y = (loss_D_real_Y + loss_D_fake_Y) * 0.5

        loss_D = loss_D_X + loss_D_Y
        disc_losses.append(loss_D.item())

        loss_D.backward()
        optimizer_D.step()

        pbar_gan.set_postfix(Loss_G=f"{loss_G.item():.4f}", Loss_D=f"{loss_D.item():.4f}")

    avg_g_loss = np.mean(gen_losses) if gen_losses else 0
    avg_d_loss = np.mean(disc_losses) if disc_losses else 0
    print(f"End of GAN Epoch {epoch_gan+1}/{num_epochs_gan} - Avg Loss G: {avg_g_loss:.4f}, Avg Loss D: {avg_d_loss:.4f}")

print("--- Finished CycleGAN Training ---")
# Keep generator_XY on DEVICE for later use
generator_XY.eval() # Set generator to eval mode for use in augmentation

# ==================================
# === Custom Classifier Model ===
# ==================================
# (Keep Mish, SEBlock, and Sentinel2Classifier classes as defined previously)
# --- Mish Activation Function ---
class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

# --- Squeeze-and-Excitation Block ---
class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1) # Global Average Pooling
        self.excitation = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            Mish(), # Using Mish activation
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
            nn.Sigmoid() # Output scaling factors between 0 and 1
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x) # Squeeze: (b, c, 1, 1)
        y = self.excitation(y) # Excitation: (b, c, 1, 1) -> (b, c // r, 1, 1) -> (b, c, 1, 1)
        return x * y # Scale original features x

# --- Sentinel-2 Classifier ---
class Sentinel2Classifier(nn.Module):
    """ Custom CNN for Sentinel-2 imagery classification. """
    def __init__(self, num_classes: int, input_channels: int = 12):
        super().__init__()
        print(f"Initializing Sentinel2Classifier with {num_classes} classes.")

        self.mish = Mish()

        # Helper to create a convolutional block
        def conv_block(in_c, out_c, kernel_size=3, stride=1, padding=1, use_se=True, reduction=16):
            layers = [
                nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding, bias=False), # No bias needed before BN
                nn.BatchNorm2d(out_c),
                Mish() # Using Mish activation
            ]
            if use_se:
                layers.append(SEBlock(out_c, reduction=reduction))
            return nn.Sequential(*layers)

        # --- Feature Extractor ---
        self.layer1 = conv_block(input_channels, 64)
        self.layer2 = conv_block(64, 128)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Downsample H, W

        self.layer3 = conv_block(128, 256)
        self.layer4 = conv_block(256, 512)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer5 = conv_block(512, 1024, use_se=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Pooling & Flatten ---
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # --- Classifier Head ---
        fc_input_features = 1024 * 2 # Concatenating avg + max pool

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(fc_input_features, 512),
            nn.BatchNorm1d(512),
            Mish(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            Mish(),
            nn.Linear(256, num_classes)
        )

        self.apply(self._initialize_weights)

    def _initialize_weights(self, m):
        """ Initializes Conv and Linear layers using Kaiming He. """
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool1(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool2(x)
        x = self.layer5(x)
        x = self.pool3(x)
        avg_pooled = self.avg_pool(x)
        max_pooled = self.max_pool(x)
        pooled = torch.cat((avg_pooled, max_pooled), dim=1)
        out = self.classifier(pooled)
        return out


# ==================================
# === Training & Validation Helper (MODIFIED for GAN Augmentation) ===
# ==================================
def run_epoch(
    model: nn.Module, # The classifier model
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Optional[GradScaler],
    device: torch.device,
    is_training: bool,
    epoch_num: int,
    num_epochs_total: int,
    current_lr: float,
    gradient_accumulation_steps: int = 1,
    warmup_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    step_warmup_per_batch: bool = True,
    # --- GAN Augmentation Args ---
    generator_for_augmentation: Optional[nn.Module] = None, # Pass the trained generator_XY
    gan_augmentation_prob: float = 0.0 # Probability to use GAN augmentation
    ):
    """Runs a single epoch of training or validation, with optional GAN augmentation."""
    if is_training:
        model.train() # Set classifier to train mode
        if optimizer is None: raise ValueError("Optimizer must be provided for training.")
        # Set generator to eval mode if provided for augmentation
        if generator_for_augmentation is not None:
            generator_for_augmentation.eval()
        context = torch.enable_grad()
        loader_desc = "Training"
    else:
        model.eval() # Set classifier to eval mode
        context = torch.no_grad()
        loader_desc = "Validation"

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    start_time = time.time()
    accum_steps = gradient_accumulation_steps if is_training else 1
    batches_augmented = 0 # Counter for augmented batches

    if is_training:
        optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch_num}/{num_epochs_total} ({loader_desc}) LR: {current_lr:.2e}")

    for batch_idx, batch_data in pbar:
        if batch_data is None or batch_data[0] is None: continue
        inputs_original, targets, _ = batch_data

        if inputs_original.numel() == 0 or targets.numel() == 0: continue

        inputs_original = inputs_original.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # --- Apply GAN Augmentation (Training Only) ---
        inputs_to_use = inputs_original # Default to original inputs
        if is_training and generator_for_augmentation is not None and random.random() < gan_augmentation_prob:
            with torch.no_grad(): # No gradients needed for generator during augmentation
                with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=CONFIG["amp_enabled"]):
                     # Generate augmented inputs using the provided generator (e.g., generator_XY)
                     inputs_augmented = generator_for_augmentation(inputs_original)
                     # *** Normalization WARNING ***
                     # The output 'inputs_augmented' is likely Tanh [-1, 1] scaled.
                     # The classifier expects per-image Z-score normalized data.
                     # This mismatch might negatively impact performance.
                     # A proper solution would involve aligning normalization.
                     # For this example, we use the direct output.
                     inputs_to_use = inputs_augmented
                     batches_augmented += 1

        # --- Classifier Forward Pass ---
        with context: # Enable/disable gradients based on is_training
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=CONFIG["amp_enabled"]):
                outputs = model(inputs_to_use) # Feed either original or augmented inputs
                loss = criterion(outputs, targets)
                if is_training:
                    loss = loss / accum_steps

        # --- Backward Pass and Optimization (Training Only) ---
        if is_training:
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(loader):
                if scaler: scaler.unscale_(optimizer)
                # Optional: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if warmup_scheduler is not None and step_warmup_per_batch:
                    warmup_scheduler.step()

        # --- Loss and Accuracy Calculation ---
        running_loss += loss.item() * inputs_original.size(0) * accum_steps # Use original size for stats
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == targets).sum().item()
        total_samples += targets.size(0)

        pbar.set_postfix(Loss=f"{running_loss / total_samples:.4f}" if total_samples > 0 else 0.0,
                         Acc=f"{correct_predictions / total_samples:.4f}" if total_samples > 0 else 0.0,
                         Augmented=f"{batches_augmented}" if is_training else "N/A")

        if is_training and (torch.isnan(loss) or torch.isinf(loss)):
            print(f"\nWARNING: NaN/Inf loss detected at Epoch {epoch_num} Batch {batch_idx+1}. Loss: {loss.item()}.")
            optimizer.zero_grad(set_to_none=True)

    # --- End of Epoch ---
    if is_training and warmup_scheduler is not None and not step_warmup_per_batch:
        warmup_scheduler.step()

    epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
    epoch_acc = correct_predictions / total_samples if total_samples > 0 else 0.0
    epoch_duration = time.time() - start_time
    augmented_info = f"({batches_augmented} batches augmented)" if is_training and batches_augmented > 0 else ""

    print(f'    Epoch {epoch_num} ({loader_desc}) Summary -> Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f} {augmented_info} | Duration: {epoch_duration:.2f}s')
    return epoch_loss, epoch_acc


# ==================================
# === Evaluation & Prediction Function ===
# ==================================
# (Keep evaluate_and_predict function as defined previously - it doesn't use GAN)
def evaluate_and_predict(
    config: Dict[str, Any],
    pred_loader: Optional[DataLoader], # Prediction loader (uses .npy files)
    device: torch.device,
    class_to_idx_map: Dict[str, int],
    model_path: Path # Path to the trained model weights
    ):
    """Loads the best model, runs prediction on the prediction dataset, saves CSV."""

    if pred_loader is None:
         print("Skipping evaluation and prediction as prediction loader is not available.")
         return

    if not model_path.exists():
         print(f"Error: Model weights not found at {model_path}. Cannot evaluate.")
         return

    num_classes = config["model"]["num_classes"]
    if num_classes is None:
         print("Error: Number of classes not set in config. Cannot initialize model.")
         return

    # Initialize model architecture
    model = Sentinel2Classifier(num_classes=num_classes).to(device)

    try:
        # Load the saved state dict
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Successfully loaded model weights from {model_path}")
    except Exception as e:
        print(f"Error loading model state_dict from {model_path}: {e}")
        traceback.print_exc()
        return # Cannot proceed without loaded weights

    model.eval() # Set model to evaluation mode
    predictions = []
    image_ids = [] # To store the base name of the image files

    with torch.no_grad():
        pbar_pred = tqdm(pred_loader, desc="Predicting on evaluation set")
        for batch_data in pbar_pred:
            if batch_data is None or batch_data[0] is None:
                continue # Skip invalid batches

            images, _, paths = batch_data # Unpack, ignore dummy label

            if images.numel() == 0: continue # Skip empty batches

            images = images.to(device, non_blocking=True)

            with torch.amp.autocast(dtype=torch.float16, enabled=config["amp_enabled"]):
                outputs = model(images)

            _, predicted_indices = torch.max(outputs.data, 1)

            predictions.extend(predicted_indices.cpu().numpy())
            image_ids.extend([Path(p).stem for p in paths])

    if not predictions:
         print("No predictions were generated. Check the prediction dataset and loader.")
         return

    idx_to_class = {v: k for k, v in class_to_idx_map.items()}
    try:
        predicted_classes = [idx_to_class[i] for i in predictions]
    except KeyError as e:
         print(f"Error: Predicted index {e} not found in idx_to_class map.")
         return

    df = pd.DataFrame({'test_id': image_ids, 'label': predicted_classes})

    csv_path = config["prediction"]["predictions_csv_path"]
    try:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df['test_id'] = df['test_id'].str.replace('test_', '', regex=False)

        df.to_csv(csv_path, index=False)
        print(f"Predictions saved successfully to {csv_path}")
    except Exception as e:
        print(f"Error saving predictions to CSV {csv_path}: {e}")

    if config["prediction"]["submit_to_kaggle"]:
        print("Kaggle submission logic placeholder.")


# ==================================
# === Main Execution Logic (MODIFIED) ===
# ==================================
if __name__ == '__main__':

    output_dir = CONFIG["model"]["load_path"].parent
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    if CONFIG["model"]["num_classes"] is None:
        raise ValueError("Number of classes not determined. Cannot proceed.")

    # --- Classifier Model Setup ---
    print("\n--- Initializing Sentinel-2 Classifier Model ---")
    classifier_model = Sentinel2Classifier(num_classes=CONFIG["model"]["num_classes"])
    classifier_model.to(DEVICE)
    print(f"Classifier initialized with {sum(p.numel() for p in classifier_model.parameters() if p.requires_grad)} trainable parameters.")

    # --- Loss for Classifier ---
    classifier_criterion = nn.CrossEntropyLoss()

    # --- Training Parameters ---
    num_epochs_total = CONFIG["train"]["stage1_epochs"] + CONFIG["train"]["stage2_epochs"]
    stage1_epochs = CONFIG["train"]["stage1_epochs"]
    stage2_epochs = CONFIG["train"]["stage2_epochs"]
    lr_stage1 = CONFIG["train"]["lr_stage1"]
    lr_stage2 = CONFIG["train"]["lr_stage2"]
    warmup_epochs = CONFIG["train"]["warmup_epochs"]
    initial_warmup_lr = CONFIG["train"]["initial_warmup_lr"]
    patience = CONFIG["train"]["patience"]
    model_save_path = CONFIG["model"]["load_path"]
    weight_decay = CONFIG["train"]["weight_decay"]
    gradient_accumulation_steps = CONFIG["train"]["gradient_accumulation_steps"]
    # GAN Augmentation parameters
    use_gan_augmentation = CONFIG["train"]["use_gan_augmentation"]
    gan_augmentation_prob = CONFIG["train"]["gan_augmentation_prob"] if use_gan_augmentation else 0.0

    # --- Training State Variables ---
    overall_best_val_loss = float('inf')
    best_model_state_dict = None
    total_epochs_run = 0
    scaler = GradScaler(enabled=CONFIG["amp_enabled"])
    print(f"AMP Enabled: {CONFIG['amp_enabled']}")
    print(f"Gradient Accumulation Steps: {gradient_accumulation_steps}")
    if use_gan_augmentation:
        print(f"GAN Augmentation Enabled with probability: {gan_augmentation_prob:.2f}")
        print(f"WARNING: GAN output normalization likely mismatches classifier input normalization. Results may vary.")
    else:
        print("GAN Augmentation Disabled.")
        generator_XY = None # Set generator to None if not used

    # === CLASSIFIER TRAINING STAGES ===

    # ============ Stage 1: Initial Training ============
    print(f"\n--- Classifier Stage 1: Initial Training ({stage1_epochs} epochs) ---")
    stage = 1

    optimizer_cls = torch.optim.AdamW(classifier_model.parameters(), lr=lr_stage1, weight_decay=weight_decay)
    print(f"Stage 1 Optimizer: AdamW, LR={lr_stage1}, WeightDecay={weight_decay}")

    # Warmup Scheduler
    warmup_scheduler_cls = None
    if warmup_epochs > 0:
        effective_batches_per_epoch = len(train_loader) // gradient_accumulation_steps
        total_warmup_steps = effective_batches_per_epoch * warmup_epochs
        if total_warmup_steps > 0:
            def lr_lambda_warmup(current_step):
                 if current_step < total_warmup_steps:
                     lr_scale = initial_warmup_lr / lr_stage1
                     progress = float(current_step) / float(max(1, total_warmup_steps))
                     return lr_scale + (1.0 - lr_scale) * progress
                 return 1.0
            warmup_scheduler_cls = torch.optim.lr_scheduler.LambdaLR(optimizer_cls, lr_lambda=lr_lambda_warmup)
            print(f"Using Linear Warmup for {warmup_epochs} epochs ({total_warmup_steps} steps)")
            step_warmup_per_batch = True
        else:
            warmup_epochs = 0 # Disable if steps are 0

    # Plateau Scheduler
    scheduler_cls = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_cls, mode='min', factor=0.1, patience=patience // 2, verbose=True
    )
    print(f"Stage 1 Plateau Scheduler: Patience={patience // 2}")
    epochs_without_improvement_stage1 = 0

    for epoch in range(stage1_epochs):
        epoch_num = total_epochs_run + 1
        current_lr = optimizer_cls.param_groups[0]['lr']

        # Run training epoch with optional GAN augmentation
        train_loss, train_accuracy = run_epoch(
            classifier_model, train_loader, classifier_criterion, optimizer_cls, scaler, DEVICE, True,
            epoch_num, num_epochs_total, current_lr, gradient_accumulation_steps,
            warmup_scheduler_cls if epoch < warmup_epochs else None,
            step_warmup_per_batch=step_warmup_per_batch,
            generator_for_augmentation=generator_XY, # Pass trained generator
            gan_augmentation_prob=gan_augmentation_prob # Pass probability
        )

        # Run validation epoch (no GAN augmentation here)
        avg_val_loss, val_accuracy = run_epoch(
            classifier_model, val_loader_split, classifier_criterion, None, scaler, DEVICE, False,
            epoch_num, num_epochs_total, current_lr, gradient_accumulation_steps,
            generator_for_augmentation=None, gan_augmentation_prob=0.0 # No aug in validation
        )

        print(f'---> Epoch {epoch_num}/{num_epochs_total} (Stage 1) | Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f} | Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}')

        if epoch >= warmup_epochs: scheduler_cls.step(avg_val_loss)

        if avg_val_loss < overall_best_val_loss:
            overall_best_val_loss = avg_val_loss
            epochs_without_improvement_stage1 = 0
            best_model_state_dict = classifier_model.state_dict()
            try:
                 torch.save(best_model_state_dict, model_save_path)
                 print(f'---> Validation Loss Improved. Model saved.')
            except Exception as e: print(f"Error saving model: {e}")
        else:
            epochs_without_improvement_stage1 += 1
            print(f'---> Val loss did not improve for {epochs_without_improvement_stage1} epochs.')
            if epochs_without_improvement_stage1 >= patience:
                print(f'!!! Early stopping triggered during Stage 1.')
                break

        total_epochs_run = epoch_num

    print(f"--- Finished Stage 1 Training ---")
    print(f"Best validation loss after Stage 1: {overall_best_val_loss:.4f}")

    # ============ Stage 2: Fine-tuning ============
    if stage2_epochs > 0:
        print(f"\n--- Classifier Stage 2: Fine-tuning All Layers ({stage2_epochs} epochs) ---")
        stage = 2

        if best_model_state_dict:
            print("Loading best model state from Stage 1.")
            classifier_model.load_state_dict(best_model_state_dict)
        else:
            print("Warning: No best model state from Stage 1.")
            classifier_model.to(DEVICE)

        optimizer_cls = torch.optim.AdamW(classifier_model.parameters(), lr=lr_stage2, weight_decay=weight_decay)
        print(f"Stage 2 Optimizer: AdamW, LR={lr_stage2}")

        scheduler_cls = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_cls, mode='min', factor=0.1, patience=patience, verbose=True
        )
        print(f"Stage 2 Plateau Scheduler: Patience={patience}")
        epochs_without_improvement_stage2 = 0

        for epoch in range(stage2_epochs):
            epoch_num = total_epochs_run + 1
            current_lr = optimizer_cls.param_groups[0]['lr']

            # Run training epoch with optional GAN augmentation
            train_loss, train_accuracy = run_epoch(
                classifier_model, train_loader, classifier_criterion, optimizer_cls, scaler, DEVICE, True,
                epoch_num, num_epochs_total, current_lr, gradient_accumulation_steps,
                warmup_scheduler=None, # No warmup in stage 2
                generator_for_augmentation=generator_XY,
                gan_augmentation_prob=gan_augmentation_prob
            )

            # Run validation epoch
            avg_val_loss, val_accuracy = run_epoch(
                classifier_model, val_loader_split, classifier_criterion, None, scaler, DEVICE, False,
                epoch_num, num_epochs_total, current_lr, gradient_accumulation_steps,
                generator_for_augmentation=None, gan_augmentation_prob=0.0
            )

            print(f'---> Epoch {epoch_num}/{num_epochs_total} (Stage 2) | Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f} | Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}')

            scheduler_cls.step(avg_val_loss)

            if avg_val_loss < overall_best_val_loss:
                overall_best_val_loss = avg_val_loss
                epochs_without_improvement_stage2 = 0
                best_model_state_dict = classifier_model.state_dict()
                try:
                     torch.save(best_model_state_dict, model_save_path)
                     print(f'---> Validation Loss Improved. Model saved.')
                except Exception as e: print(f"Error saving model: {e}")
            else:
                epochs_without_improvement_stage2 += 1
                print(f'---> Val loss did not improve for {epochs_without_improvement_stage2} epochs.')
                if epochs_without_improvement_stage2 >= patience:
                    print(f'!!! Early stopping triggered during Stage 2.')
                    break

            total_epochs_run = epoch_num
        print(f"--- Finished Stage 2 Training ---")

    # --- Final Report ---
    print(f"\n--- Classifier Training Complete ---")
    print(f"Total classifier epochs run: {total_epochs_run}")
    if best_model_state_dict is not None:
        print(f'Best overall validation loss: {overall_best_val_loss:.4f}')
        print(f'Best classifier model weights saved to: {model_save_path}')
    else:
        print("Training completed, but no improvement seen or no model saved.")

    # --- Final Evaluation ---
    print("\n--- Running Final Evaluation on Prediction Set ---")
    evaluate_and_predict(
        config=CONFIG,
        pred_loader=final_pred_loader,
        device=DEVICE,
        class_to_idx_map=class_to_idx_map,
        model_path=model_save_path # Use the best saved classifier model
    )

    print("\n--- Script Finished ---")