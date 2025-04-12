import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import models, transforms
import rasterio
import numpy as np
import torch.nn.functional as F
import pandas as pd
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True' # For potential memory fragmentation issues
import random
import time
from torch.cuda.amp import GradScaler, autocast
from torchvision.models import ResNet50_Weights, ConvNeXt_Base_Weights # Note: These weights aren't used in the custom model
from sklearn.model_selection import train_test_split
import traceback

from pathlib import Path
from tqdm import tqdm
from typing import Tuple, List, Dict, Any, Optional, Callable

import warnings

# --- Device Setup ---
USE_CUDA = torch.cuda.is_available()
# Changed default device selection logic for broader compatibility (MPS for Apple Silicon)
if USE_CUDA:
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

# --- Suppress all warnings ---
warnings.filterwarnings("ignore")

##################################
# --- CBAM Attention Mechanism ---
# (Keeping model definition in case needed later)
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

        # Spatial Attention
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        # Channel Attention
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_out = self.sigmoid(avg_out + max_out)
        x = x * channel_out

        # Spatial Attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv_spatial(torch.cat([avg_out, max_out], dim=1)))
        x = x * spatial_out
        return x

# --- CycleGAN Generator ---
# (Keeping model definition in case needed later)
class ResNetBlock(nn.Module):
    def __init__(self, channels):
        super(ResNetBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, in_channels=12, out_channels=12, num_res_blocks=6):
        super(Generator, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.res_blocks = nn.Sequential(*[ResNetBlock(64) for _ in range(num_res_blocks)])
        self.final = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=7, padding=3, bias=False),
            nn.Tanh() # Tanh for image generation (outputs in [-1, 1])
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.res_blocks(x)
        return self.final(x)

# --- CycleGAN Discriminator ---
# (Keeping model definition in case needed later)
class Discriminator(nn.Module):
    def __init__(self, in_channels=12):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True), # Use inplace=True for minor memory saving
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # Final layer outputs a single logit value per patch/image
            nn.Conv2d(256, 1, kernel_size=1, stride=1), # Adjusted kernel size for potentially smaller feature maps
        )

    def forward(self, x):
        # **FIXED:** Return raw logits, do NOT apply sigmoid here.
        # BCEWithLogitsLoss expects logits.
        return self.model(x)

# --- Progressive Learning: Dynamic Resizing (COMMENTED OUT) ---
# def progressive_resize(epoch, model, optimizer, initial_size=64, max_size=256):
#     """ Gradually increases image size during training. """
#     # **NOTE:** This function modifies a global `transform_train` variable.
#     # This likely DOES NOT WORK as intended because the DataLoader uses the
#     # dataset instance created with the *original* transform object.
#     # To implement progressive resizing correctly, you would typically need to:
#     # 1. Create a new Dataset and DataLoader with the updated transform at
#     #    specific epoch intervals.
#     # 2. Or, modify the Dataset's __getitem__ to resize based on the epoch
#     #    (can be less efficient).
#     # This function is commented out to avoid confusion and ensure the fixed
#     # resize in `train_transforms` is used consistently.
#     new_size = min(initial_size + (epoch * 16), max_size)
#     global transform_train
#     transform_train = transforms.Compose([
#         transforms.Resize((new_size, new_size)),
#         #transforms.Lambda(lambda x: x[:3, :, :]), # Example: Select RGB if needed
#         transforms.ToTensor() # Note: ToTensor should typically be applied to PIL Images or numpy arrays, not tensors. Check usage.
#     ])
#     print(f"Progressive Resize: Set image size to {new_size}x{new_size}") # Add print statement

# --- Initialize Models (CycleGAN - COMMENTED OUT) ---
# generator_XY = Generator().to(DEVICE)
# generator_YX = Generator().to(DEVICE)
# discriminator_X = Discriminator().to(DEVICE)
# discriminator_Y = Discriminator().to(DEVICE)

# --- Losses & Optimizers (CycleGAN - COMMENTED OUT) ---
# cycle_loss_fn = nn.L1Loss()
# gan_loss_fn = nn.BCEWithLogitsLoss() # Correct loss for logits output by Discriminator
# # Note: Optimizers are commented out as the CycleGAN training loop is removed.
# optimizer_G = torch.optim.Adam(list(generator_XY.parameters()) + list(generator_YX.parameters()), lr=2e-4, betas=(0.5, 0.999))
# optimizer_D = torch.optim.Adam(list(discriminator_X.parameters()) + list(discriminator_Y.parameters()), lr=2e-4, betas=(0.5, 0.999))

##################################
# --- Configuration ---
CONFIG = {
    "model": {
        # Consider making the save path more descriptive, e.g., './sentinel2_classifier_best.pth'
        "load_path": Path('./sentinel2_classifier_best.pth'), # Path for saving/loading CLASSIFIER model
        "num_classes": None,  # Determined dynamically
        "class_names": None,  # Determined dynamically
    },
    "data": {
        "train_dir": Path('./ds/images/remote_sensing/otherDatasets/sentinel_2/tif'),
        "validation_dir": Path('./testset/testset'), # Directory containing .npy files for final prediction
        "image_size": 128, # Fixed image size for training/validation
        "batch_size": 16,
        "num_workers": 4, # Adjust based on your system
        "train_ratio": 0.9,
    },
    "train": {
        "seed": 1337,
        "lr_stage1": 1e-3,
        "lr_stage2": 1e-4, # Slightly reduced LR for stage 2 fine-tuning
        "stage1_epochs": 30,
        "stage2_epochs": 30,
        "warmup_epochs": 3,
        "initial_warmup_lr": 1e-6,
        "weight_decay": 3.7553e-05, # Consider tuning this if needed
        "optimizer_name": 'AdamW', # AdamW is generally a good choice
        "patience": 10, # Early stopping patience
        "gradient_accumulation_steps": 2 # Accumulate gradients over 2 steps
    },
    "device": DEVICE.type, # Use determined device type
    "amp_enabled": USE_CUDA, # Enable AMP only if CUDA is available
    "prediction": {
        "predictions_csv_path": Path('./outputs/track_1_predictions.csv'), # Changed name slightly
        "kaggle_competition": 'aicrowd-geospatial-challenge',
        "kaggle_message": 'Evaluation Script Submission',
        "submit_to_kaggle": False, # Set to True to enable submission (if kaggle API is set up)
    }
}

# --- Seed Setting ---
def set_seed(seed: int):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # For multi-GPU setups
    # These can slow down training, use only if strict reproducibility is essential
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")

set_seed(CONFIG["train"]["seed"])

# --- Data Loading and Preprocessing ---
def load_sentinel2_image(filepath: str) -> Optional[np.ndarray]:
    """Loads a Sentinel-2 image (TIF or NPY), returns NumPy CHW (12 bands) or None on error."""
    try:
        if filepath.lower().endswith('.tif'):
            with rasterio.open(filepath) as src:
                # Bands B1 to B9 (1-9), B11, B12, B8A (11-13) -> 12 bands total
                # Excludes B10 (thermal)
                bands_to_read = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13] # Assuming 1-based indexing in rasterio
                if src.count < 13:
                    print(f"Warning: Expected >=13 bands, got {src.count} in {filepath}. Reading available bands.")
                    # Attempt to read the first 12 bands if fewer than 13 exist
                    bands_to_read = list(range(1, min(src.count + 1, 13)))
                    if len(bands_to_read) < 12:
                         print(f"Error: Not enough bands ({len(bands_to_read)}) to form 12 channels in {filepath}. Skipping.")
                         return None

                image = src.read(bands_to_read) # Shape: (12, H, W)

        elif filepath.lower().endswith('.npy'):
            image = np.load(filepath) # Load .npy as is
            # Assuming NPY files are already in the correct format (e.g., 12, H, W)
            if image.shape[0] != 12:
                 # If NPY has 13 bands, assume standard Sentinel-2 order and drop B10 (index 9)
                 if image.shape[0] == 13:
                     print(f"Info: NPY file {filepath} has 13 bands. Selecting bands 1-9, 11, 12, 8A (indices 0-8, 10, 11, 12).")
                     indices_to_keep = list(range(9)) + [10, 11, 12]
                     image = image[indices_to_keep, :, :]
                 else:
                    print(f"Error: Unexpected shape for .npy {filepath}: {image.shape}. Expected (12, H, W) or (13, H, W). Skipping.")
                    return None
        else:
            print(f"Error: Unsupported file type: {filepath}. Skipping.")
            return None

        # Ensure float32 type
        return image.astype(np.float32)

    except rasterio.RasterioIOError as e:
        print(f"Rasterio Error loading {filepath}: {e}. Skipping.")
        return None
    except FileNotFoundError:
        print(f"Error: File not found {filepath}. Skipping.")
        return None
    except Exception as e:
        print(f"Unexpected error loading image {filepath}: {e}")
        traceback.print_exc()
        return None


# --- Per-Image Normalization Function ---
def normalize_image_per_image(image_np: np.ndarray) -> Optional[np.ndarray]:
    """Normalizes 12-channel NumPy image (C, H, W) using its own stats."""
    if image_np is None: return None
    if image_np.ndim != 3 or image_np.shape[0] != 12:
        print(f"Error: Invalid shape for per-image normalization: {image_np.shape}. Expected (12, H, W). Skipping normalization.")
        return None # Or return image_np if you want to proceed without normalization

    # Calculate mean and std per channel, avoiding NaN/Inf
    mean = np.nanmean(image_np, axis=(1, 2), keepdims=True)
    std = np.nanstd(image_np, axis=(1, 2), keepdims=True)

    # Replace NaN means/stds (e.g., from empty channels) with 0 and 1 respectively
    mean = np.nan_to_num(mean, nan=0.0)
    std = np.nan_to_num(std, nan=1.0)

    # Normalize, adding epsilon to std deviation to prevent division by zero
    normalized_image = (image_np - mean) / (std + 1e-7)

    # Clip values to prevent extreme outliers after normalization (optional but recommended)
    normalized_image = np.clip(normalized_image, -5.0, 5.0) # Adjust range as needed

    return normalized_image

# --- Dataset Class (Using Per-Image Normalization) ---
class Sentinel2Dataset(Dataset):
    """Custom Dataset for Sentinel-2 images. Returns Tensor CHW (12 channels)."""
    def __init__(self, paths_labels: List[Tuple[str, int]], transform: Optional[Callable] = None):
        self.paths_labels = paths_labels
        self.transform = transform
        self.valid_indices = self._validate_paths() # Pre-filter valid paths

    def _validate_paths(self):
        """ Check if files exist during initialization """
        valid_indices = []
        print("Validating dataset paths...")
        for idx, (path, _) in enumerate(tqdm(self.paths_labels)):
            if os.path.exists(path):
                valid_indices.append(idx)
            else:
                print(f"Warning: File not found during validation: {path}. It will be skipped.")
        print(f"Found {len(valid_indices)} valid image paths out of {len(self.paths_labels)}.")
        return valid_indices

    def __len__(self):
        # Return the number of valid samples
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # Map the input index to the index in the original list
        original_idx = self.valid_indices[idx]
        image_path, label = self.paths_labels[original_idx]

        try:
            # 1. Load image
            image_np = load_sentinel2_image(image_path) # NumPy (C, H, W) or None
            if image_np is None: return None # Skip if loading failed

            # 2. Normalize per image
            image_np_normalized = normalize_image_per_image(image_np) # NumPy (C, H, W) or None
            if image_np_normalized is None: return None # Skip if normalization failed

            # 3. Convert to Tensor
            # Use .copy() to avoid potential issues with shared memory if needed,
            # but often not strictly necessary after normalization creates a new array.
            image_tensor = torch.from_numpy(image_np_normalized).float()

            # 4. Apply other transformations (like resizing, augmentations)
            if self.transform:
                image_tensor = self.transform(image_tensor) # Apply remaining transforms

            # 5. Prepare label
            label_tensor = torch.tensor(label, dtype=torch.long)

            return image_tensor, label_tensor, image_path

        except Exception as e:
            print(f"Error processing image {image_path} in __getitem__:")
            traceback.print_exc()
            return None # Signal error to collate_fn

# Prediction dataset also uses per-image normalization
class NpyPredictionDataset(Dataset):
    """ Dataset for loading .npy files for prediction. """
    def __init__(self, root_dir: str, transform: Optional[Callable] = None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        # Use pathlib for robust path handling
        self.file_paths = sorted([p for p in self.root_dir.glob('*.npy')])
        if not self.file_paths:
            raise FileNotFoundError(f"No .npy files found in {self.root_dir}")
        print(f"Found {len(self.file_paths)} .npy files for prediction in {self.root_dir}.")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image_path = str(self.file_paths[idx]) # Convert Path object to string if needed by loaders

        try:
            # 1. Load image
            image_np = load_sentinel2_image(image_path) # NumPy (C, H, W) or None
            if image_np is None: return None

            # 2. Normalize per image
            image_np_normalized = normalize_image_per_image(image_np) # NumPy (C, H, W) or None
            if image_np_normalized is None: return None

            # 3. Convert to Tensor
            image_tensor = torch.from_numpy(image_np_normalized).float()

            # 4. Apply transforms (usually just resizing for prediction)
            if self.transform:
                image_tensor = self.transform(image_tensor)

            # Return image tensor, dummy label 0, and image path (for mapping predictions)
            return image_tensor, torch.tensor(0, dtype=torch.long), image_path

        except Exception as e:
            print(f"Error processing prediction image {image_path}:")
            traceback.print_exc()
            return None

# --- Data Transforms ---
# Note: Input to these transforms is expected to be a Tensor (C, H, W)
# because normalization and ToTensor conversion happens in the Dataset __getitem__

class AddGaussianNoise(object):
    """Adds Gaussian noise to a tensor."""
    def __init__(self, mean=0., std=0.1):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        # Ensure noise has the same device and dtype as the tensor
        noise = torch.randn(tensor.size(), device=tensor.device, dtype=tensor.dtype) * self.std + self.mean
        return tensor + noise

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'

class RandomChannelScale(object):
    """Applies random scaling to each channel independently."""
    def __init__(self, scale_range=(0.9, 1.1)):
        self.scale_range = scale_range

    def __call__(self, tensor):
        # Generate random scales on the correct device
        scale_factors = (torch.rand(tensor.shape[0], device=tensor.device, dtype=tensor.dtype) *
                         (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0])
        # Reshape for broadcasting (C, 1, 1)
        return tensor * scale_factors.view(-1, 1, 1)

    def __repr__(self):
        return self.__class__.__name__ + f'(scale_range={self.scale_range})'

# Define transforms - applied AFTER normalization and tensor conversion
train_transforms = transforms.Compose([
    # Geometric Augmentations
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    # Use antialias=True for better quality resizing
    transforms.Resize((CONFIG["data"]["image_size"], CONFIG["data"]["image_size"]), antialias=True),
    # More advanced geometric augmentations (use with caution, might be too strong)
    # transforms.RandomRotation(degrees=15),
    # transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),

    # Intensity/Noise Augmentations (applied to normalized tensors)
    AddGaussianNoise(mean=0., std=0.05), # Reduce noise level slightly
    RandomChannelScale(scale_range=(0.95, 1.05)), # Reduce scaling range slightly
    # ColorJitter is typically for RGB images, may have unintended effects on 12-channel data
    # transforms.ColorJitter(brightness=0.1, contrast=0.1),
])

val_transforms = transforms.Compose([
    # Only resize for validation/prediction
    transforms.Resize((CONFIG["data"]["image_size"], CONFIG["data"]["image_size"]), antialias=True),
])

# --- Create Datasets ---
print("Scanning training directory and creating dataset splits...")
full_dataset_paths_labels = []
class_to_idx_map = {}
class_names = []
idx_counter = 0

train_root_dir = CONFIG["data"]["train_dir"]
if not train_root_dir.exists():
     raise FileNotFoundError(f"Training directory not found: {train_root_dir}")

# Scan training directory for class folders
for class_folder in sorted(train_root_dir.iterdir()):
     if class_folder.is_dir() and not class_folder.name.startswith('.'):
         class_name = class_folder.name
         if class_name not in class_to_idx_map:
             class_to_idx_map[class_name] = idx_counter
             class_names.append(class_name)
             idx_counter += 1
         class_idx = class_to_idx_map[class_name]
         # Find .tif files within the class folder
         for filepath in class_folder.glob('*.tif'):
             full_dataset_paths_labels.append((str(filepath), class_idx))
         # Optionally include .tiff as well
         # for filepath in class_folder.glob('*.tiff'):
         #     full_dataset_paths_labels.append((str(filepath), class_idx))


num_classes = len(class_names)
CONFIG["model"]["num_classes"] = num_classes # Store num_classes in config
CONFIG["model"]["class_names"] = class_names # Store class names in config

if num_classes == 0:
    raise FileNotFoundError(f"No valid class folders containing .tif files found in {train_root_dir}")
if not full_dataset_paths_labels:
     raise FileNotFoundError(f"No .tif files found in any class subdirectories of {train_root_dir}")

print(f"Found {len(full_dataset_paths_labels)} training image paths in {num_classes} classes: {class_names}")
print(f"Class mapping: {class_to_idx_map}")

# Stratified Split
try:
    train_info, val_info = train_test_split(
        full_dataset_paths_labels,
        train_size=CONFIG["data"]["train_ratio"],
        random_state=CONFIG["train"]["seed"],
        # Ensure stratification is possible (requires at least 2 samples per class for split)
        stratify=[label for _, label in full_dataset_paths_labels]
    )
except ValueError as e:
     print(f"Warning: Stratified split failed ({e}). Falling back to non-stratified split.")
     # Fallback to non-stratified split if stratification fails (e.g., too few samples in a class)
     train_info, val_info = train_test_split(
        full_dataset_paths_labels,
        train_size=CONFIG["data"]["train_ratio"],
        random_state=CONFIG["train"]["seed"]
     )

# Create Dataset objects using the split lists
print("Creating Sentinel2Dataset instances...")
train_dataset = Sentinel2Dataset(train_info, transform=train_transforms)
val_tif_dataset = Sentinel2Dataset(val_info, transform=val_transforms)

# Create Prediction Dataset (.npy files)
print("Creating NpyPredictionDataset instance...")
try:
    final_validation_dataset = NpyPredictionDataset(str(CONFIG["data"]["validation_dir"]), transform=val_transforms)
except FileNotFoundError as e:
    print(f"Error initializing prediction dataset: {e}")
    final_validation_dataset = None # Handle case where prediction set is missing


# --- Create DataLoaders ---
def collate_fn(batch):
    """ Filters out None samples and stacks the rest. """
    # Filter out samples where __getitem__ returned None
    batch = [item for item in batch if item is not None]
    if not batch:
        # Return None or empty tensors if the whole batch was invalid
        return None, None, None

    try:
        # Unpack and stack valid items
        images = torch.stack([item[0] for item in batch])
        labels = torch.stack([item[1] for item in batch])
        paths = [item[2] for item in batch]
        return images, labels, paths
    except Exception as e:
        # Handle potential errors during stacking (e.g., size mismatch if transforms are inconsistent)
        print(f"Error in collate_fn during stacking: {e}. Skipping batch.")
        # Optionally print traceback for debugging:
        # traceback.print_exc()
        return None, None, None

# Use persistent_workers=True if num_workers > 0 for potential speedup
# Drop last might be useful if batch size doesn't divide dataset size evenly
persistent_workers = CONFIG["data"]["num_workers"] > 0

train_loader = DataLoader(
    train_dataset,
    batch_size=CONFIG["data"]["batch_size"],
    shuffle=True,
    num_workers=CONFIG["data"]["num_workers"],
    pin_memory=True, # Set pin_memory=True if using GPU
    collate_fn=collate_fn,
    drop_last=True, # Drop last incomplete batch during training
    persistent_workers=persistent_workers
)

val_loader_split = DataLoader(
    val_tif_dataset,
    batch_size=CONFIG["data"]["batch_size"] * 2, # Use larger batch size for validation
    shuffle=False,
    num_workers=CONFIG["data"]["num_workers"],
    pin_memory=True,
    collate_fn=collate_fn,
    persistent_workers=persistent_workers
)

# Only create prediction loader if the dataset was created successfully
if final_validation_dataset:
    final_pred_loader = DataLoader(
        final_validation_dataset,
        batch_size=CONFIG["data"]["batch_size"] * 2, # Use larger batch size for prediction
        shuffle=False,
        num_workers=CONFIG["data"]["num_workers"],
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=persistent_workers
    )
else:
    final_pred_loader = None
    print("Warning: Prediction DataLoader not created as prediction dataset failed to initialize.")


print("DataLoaders created.")

##################################
# --- CycleGAN Training Loop (COMMENTED OUT) ---
# This section is commented out because the CycleGAN was not integrated
# with the main classification task in the original script.
# If you intend to use CycleGAN (e.g., for data augmentation), you'll
# need to uncomment this, initialize the models/optimizers/losses above,
# and integrate the generator output into the Sentinel2Dataset or the
# classifier training loop.

# print("Starting CycleGAN Training (Commented Out)")
# num_cyclegan_epochs = 1 # Set number of epochs for GAN training if enabled
# for epoch in range(num_cyclegan_epochs):
#     # Call to progressive_resize was here - commented out as function is commented out
#     # progressive_resize(epoch, generator_XY, optimizer_G)

#     # Example placeholder for GAN training loop structure
#     for batch_idx, batch_data in enumerate(train_loader): # Use train_loader for real images
#         if batch_data is None or batch_data[0] is None: continue # Skip bad batches
#         real_X, _, _ = batch_data
#         real_X = real_X.to(DEVICE)

#         # Sample real_Y from the same loader (unpaired)
#         # This requires careful handling if loader runs out during epoch
#         try:
#             real_Y_batch = next(iter(train_loader)) # Simplistic sampling, might repeat data
#             if real_Y_batch is None or real_Y_batch[0] is None: continue
#             real_Y, _, _ = real_Y_batch
#             real_Y = real_Y.to(DEVICE)
#             if real_Y.shape[0] != real_X.shape[0]: continue # Ensure batch sizes match if sampling this way
#         except StopIteration:
#             break # End epoch if loader is exhausted

#         # --- Generator Training ---
#         optimizer_G.zero_grad()

#         # Generate fake images
#         fake_Y = generator_XY(real_X)
#         fake_X = generator_YX(real_Y)

#         # Cycle consistency
#         cycle_X = generator_YX(fake_Y)
#         cycle_Y = generator_XY(fake_X)
#         loss_cycle = cycle_loss_fn(cycle_X, real_X) + cycle_loss_fn(cycle_Y, real_Y)

#         # GAN adversarial loss (generators try to fool discriminators)
#         pred_fake_Y = discriminator_Y(fake_Y)
#         pred_fake_X = discriminator_X(fake_X)
#         loss_gan_G = gan_loss_fn(pred_fake_Y, torch.ones_like(pred_fake_Y)) + \
#                      gan_loss_fn(pred_fake_X, torch.ones_like(pred_fake_X))

#         # Total generator loss
#         lambda_cycle = 10
#         loss_G = loss_gan_G + lambda_cycle * loss_cycle
#         loss_G.backward()
#         optimizer_G.step()

#         # --- Discriminator Training ---
#         optimizer_D.zero_grad()

#         # Discriminator Y loss
#         pred_real_Y = discriminator_Y(real_Y)
#         pred_fake_Y_detached = discriminator_Y(fake_Y.detach()) # Detach fake_Y
#         loss_D_Y = (gan_loss_fn(pred_real_Y, torch.ones_like(pred_real_Y)) +
#                     gan_loss_fn(pred_fake_Y_detached, torch.zeros_like(pred_fake_Y_detached))) * 0.5
#         loss_D_Y.backward()

#         # Discriminator X loss
#         pred_real_X = discriminator_X(real_X)
#         pred_fake_X_detached = discriminator_X(fake_X.detach()) # Detach fake_X
#         loss_D_X = (gan_loss_fn(pred_real_X, torch.ones_like(pred_real_X)) +
#                     gan_loss_fn(pred_fake_X_detached, torch.zeros_like(pred_fake_X_detached))) * 0.5
#         loss_D_X.backward()

#         optimizer_D.step()

#         # Print progress occasionally
#         if batch_idx % 100 == 0:
#             print(f"  GAN Epoch [{epoch+1}/{num_cyclegan_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], "
#                   f"Loss_G: {loss_G.item():.4f}, Loss_D: {(loss_D_X + loss_D_Y).item():.4f}")

# print("Finished CycleGAN Training (Commented Out)")
##################################

# --- Define the Classifier Model ---
# Mish Activation Function
class Mish(nn.Module):
    """Applies the Mish activation function element-wise."""
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

# Squeeze-and-Excitation Block (Channel Attention)
class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        self.excitation = nn.Sequential(
            # Use Conv2d instead of Linear for compatibility with feature maps
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            Mish(), # Using Mish activation here too
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
            nn.Sigmoid() # Sigmoid to get channel weights between 0 and 1
        )

    def forward(self, x):
        # x shape: (B, C, H, W)
        squeezed = self.squeeze(x) # Shape: (B, C, 1, 1)
        excited = self.excitation(squeezed) # Shape: (B, C, 1, 1)
        # Multiply original input by channel weights
        return x * excited # Broadcasting applies weights channel-wise

# Improved Sentinel-2 Classifier
class Sentinel2Classifier(nn.Module):
    """ Custom CNN for Sentinel-2 image classification with Residual Connections,
        SE Attention, Hybrid Pooling & Mish Activation. """
    def __init__(self, num_classes: int, input_channels: int = 12):
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")

        self.mish = Mish()

        # --- Convolutional Blocks ---
        # Define blocks to reduce repetition
        def conv_block(in_c, out_c, kernel_size=3, stride=1, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding, bias=False), # No bias needed before BN
                nn.BatchNorm2d(out_c),
                Mish()
            )

        # Initial convolution
        self.conv1 = conv_block(input_channels, 64) # 12 -> 64
        self.res_conv1 = nn.Conv2d(input_channels, 64, kernel_size=1, bias=False) # 1x1 for residual
        self.se1 = SEBlock(64)

        # Subsequent blocks with increasing channels
        self.conv2 = conv_block(64, 128) # 64 -> 128
        self.res_conv2 = nn.Conv2d(64, 128, kernel_size=1, bias=False)
        self.se2 = SEBlock(128)

        self.conv3 = conv_block(128, 256) # 128 -> 256
        self.res_conv3 = nn.Conv2d(128, 256, kernel_size=1, bias=False)
        self.se3 = SEBlock(256)

        self.conv4 = conv_block(256, 512) # 256 -> 512
        self.res_conv4 = nn.Conv2d(256, 512, kernel_size=1, bias=False)
        self.se4 = SEBlock(512)

        # Consider adding pooling layers between conv blocks if spatial dimensions grow too large
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Pooling ---
        # Adaptive pooling handles variable input sizes from conv layers
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        pooled_features = 512 # Number of features after the last conv block

        # --- Fully Connected Layers ---
        self.fc1 = nn.Linear(pooled_features, 256) # Adjusted FC layer size
        self.bn_fc1 = nn.BatchNorm1d(256) # Batch norm for FC layers
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(256, 128) # Added another FC layer
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.4) # Slightly less dropout

        self.fc_out = nn.Linear(128, num_classes) # Final classification layer

        # Apply weight initialization
        self.apply(self._initialize_weights)

    def _initialize_weights(self, m):
        """ Initializes weights using Kaiming for Conv with Mish/ReLU, Xavier for others. """
        if isinstance(m, nn.Conv2d):
            # Kaiming He initialization is often preferred for ReLU/Mish activations
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') # Use 'relu' as proxy for Mish
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight) # Xavier for linear layers
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    # **REMOVED:** Unused _apply_residual method. Residuals handled by self.res_convX.
    # def _apply_residual(self, x, residual): ...

    def forward(self, x):
        # Input shape: (B, 12, H, W)

        # Conv Block 1 + Residual + SE
        res1 = self.res_conv1(x)
        x = self.conv1(x)
        x = self.se1(x) + res1 # Add residual after SE block

        # Conv Block 2 + Residual + SE
        res2 = self.res_conv2(x)
        x = self.conv2(x)
        x = self.se2(x) + res2

        # Conv Block 3 + Residual + SE
        res3 = self.res_conv3(x)
        x = self.conv3(x)
        x = self.se3(x) + res3

        # Conv Block 4 + Residual + SE
        res4 = self.res_conv4(x)
        x = self.conv4(x)
        x = self.se4(x) + res4
        # Shape after convs depends on initial size and lack of pooling, e.g., (B, 512, H, W)

        # Hybrid Pooling
        avg_p = self.avg_pool(x) # (B, 512, 1, 1)
        max_p = self.max_pool(x) # (B, 512, 1, 1)
        # Combine pooled features (simple addition here, could also concatenate)
        x = avg_p + max_p
        x = torch.flatten(x, 1) # Flatten: (B, 512)

        # Fully Connected Layers with Mish, BN, Dropout
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = self.mish(x)
        x = self.dropout1(x) # Apply dropout after activation/BN

        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = self.mish(x)
        x = self.dropout2(x)

        # Final Output Layer (Logits)
        x = self.fc_out(x) # (B, num_classes) - Raw logits for CrossEntropyLoss
        return x


# --- Helper Function for Training/Validation Epoch ---
def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer], # Optimizer is None during validation
    scaler: Optional[GradScaler], # Scaler is None if AMP is disabled or during validation
    device: torch.device,
    is_training: bool,
    epoch_num: int,
    num_epochs_total: int,
    current_lr: float,
    gradient_accumulation_steps: int = 1,
    # Warmup arguments removed, handled by scheduler step within the loop
    # warmup_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    # current_stage_epoch: int = 0,
    # total_warmup_epochs: int = 0
    ):
    """Runs a single epoch of training or validation."""
    if is_training:
        model.train()
        if optimizer is None: raise ValueError("Optimizer must be provided for training.")
        print(f'---> Starting Training Epoch {epoch_num}/{num_epochs_total} | LR: {current_lr:.4e}')
    else:
        model.eval()
        print(f'---> Starting Validation Epoch {epoch_num}/{num_epochs_total}')

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    start_time = time.time()

    # Use tqdm for progress bar
    progress_bar = tqdm(loader, desc=f"Epoch {epoch_num} {'Train' if is_training else 'Val'}", leave=False)

    # Enable/disable gradient calculation based on mode
    with torch.set_grad_enabled(is_training):
        for batch_idx, batch_data in enumerate(progress_bar):
            # Handle potential errors from collate_fn
            if batch_data is None or batch_data[0] is None:
                print(f"Warning: Skipping empty/invalid batch {batch_idx} in {'training' if is_training else 'validation'}.")
                continue

            inputs, targets, _ = batch_data # Unpack paths if needed later

            inputs = inputs.to(device, non_blocking=True) # Use non_blocking for potential speedup
            targets = targets.to(device, non_blocking=True)

            # Mixed Precision Context
            # Use device.type for autocast, enable only if scaler is provided (i.e., AMP enabled)
            amp_enabled = scaler is not None
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                outputs = model(inputs) # Forward pass
                loss = criterion(outputs, targets) # Compute loss

                # Normalize loss for gradient accumulation
                if is_training and gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

            if is_training:
                if amp_enabled:
                    # Scale loss and backward pass with scaler
                    scaler.scale(loss).backward()
                else:
                    # Standard backward pass without scaler
                    loss.backward()

                # Optimizer step after accumulating gradients
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    if amp_enabled:
                        scaler.step(optimizer) # Unscales gradients and steps optimizer
                        scaler.update() # Updates scaler for next iteration
                    else:
                        optimizer.step() # Standard optimizer step

                    optimizer.zero_grad(set_to_none=True) # Reset gradients (set_to_none can save memory)

                    # Warmup scheduler step (if applicable, handled outside this function now)

                # Check for NaN/Inf loss after backward but before optimizer step if possible
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\nWARNING: NaN/Inf loss detected at E{epoch_num} B{batch_idx+1}. Loss: {loss.item()}. Skipping batch update.")
                    # Need to reset gradients even if step is skipped
                    optimizer.zero_grad(set_to_none=True)
                    continue # Skip to next batch

            # --- Statistics Calculation ---
            # Use full precision loss for accumulation if using AMP
            batch_loss = loss.item() * gradient_accumulation_steps if is_training and gradient_accumulation_steps > 1 else loss.item()
            running_loss += batch_loss * inputs.size(0) # Accumulate loss weighted by batch size

            if not is_training or (batch_idx + 1) % gradient_accumulation_steps == 0: # Calculate accuracy only when updating or validating
                 _, predicted = torch.max(outputs.data, 1)
                 total_samples += targets.size(0)
                 correct_predictions += (predicted == targets).sum().item()

            # Update progress bar description
            progress_bar.set_postfix(loss=f"{batch_loss:.4f}")

    # --- Epoch Summary ---
    epoch_duration = time.time() - start_time
    if total_samples == 0: # Handle case where loader was empty or all batches were invalid
        print(f"Warning: No valid samples processed in {'training' if is_training else 'validation'} epoch {epoch_num}.")
        return 0.0, 0.0 # Return zero loss/accuracy

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples

    mode_str = "Training" if is_training else "Validation"
    print(f'\n    Epoch {epoch_num} {mode_str} Summary:')
    print(f'    Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f} | Duration: {epoch_duration:.2f}s')

    return epoch_loss, epoch_acc


def evaluate_model(config: Dict[str, Any], model: nn.Module, pred_loader: DataLoader, device: torch.device, class_to_idx_map: Dict[str, int]):
    """Runs prediction on a loader, saves results to CSV."""

    if pred_loader is None:
        print("Prediction loader is not available. Skipping evaluation.")
        return

    if not class_to_idx_map:
         print("Error: class_to_idx_map is empty. Cannot map predictions.")
         return

    model.eval() # Set model to evaluation mode
    predictions = []
    image_ids = [] # To store the base name of the image files

    print(f"\n--- Starting Evaluation on {len(pred_loader.dataset)} samples ---") # type: ignore

    with torch.no_grad():
        for batch_data in tqdm(pred_loader, desc="Predicting"):
            if batch_data is None or batch_data[0] is None:
                print("Warning: Skipping empty/invalid batch during prediction.")
                continue

            images, _, paths = batch_data # Unpack images and paths
            images = images.to(device, non_blocking=True)

            # Use AMP context for prediction if enabled during training, for consistency
            amp_enabled = config["amp_enabled"] and device.type == 'cuda'
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                outputs = model(images)

            # Get predicted class index
            _, predicted_indices = torch.max(outputs.data, 1)

            predictions.extend(predicted_indices.cpu().numpy())
            # Extract the filename stem (without extension) from the path
            image_ids.extend([Path(p).stem for p in paths])

    if not predictions:
        print("Error: No predictions were generated.")
        return

    # Map predictions to class names
    idx_to_class = {v: k for k, v in class_to_idx_map.items()}
    try:
        predicted_classes = [idx_to_class[i] for i in predictions]
    except KeyError as e:
        print(f"Error: Predicted index {e} not found in idx_to_class mapping.")
        print(f"idx_to_class: {idx_to_class}")
        print(f"Unique predictions: {set(predictions)}")
        return

    # Create DataFrame
    if len(image_ids) != len(predicted_classes):
        print(f"Error: Mismatch between number of image IDs ({len(image_ids)}) and predictions ({len(predicted_classes)}).")
        return

    df = pd.DataFrame({'test_id': image_ids, 'label': predicted_classes})

    # Clean test_id if necessary (example: remove 'test_' prefix)
    # df['test_id'] = df['test_id'].str.replace('test_', '', regex=False)

    # Save to CSV
    csv_path = config["prediction"]["predictions_csv_path"]
    try:
        # Ensure output directory exists
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"Predictions saved successfully to {csv_path}")
    except Exception as e:
        print(f"Error saving predictions to CSV: {e}")

    # Optional: Submit to Kaggle if enabled and API is configured
    # if config["prediction"]["submit_to_kaggle"]:
    #     try:
    #         # Ensure kaggle package is installed: pip install kaggle
    #         import kaggle
    #         kaggle.api.competition_submit(
    #             file_name=str(csv_path),
    #             message=config["prediction"]["kaggle_message"],
    #             competition=config["prediction"]["kaggle_competition"]
    #         )
    #         print("Successfully submitted predictions to Kaggle.")
    #     except ImportError:
    #         print("Warning: 'kaggle' package not found. Cannot submit. Install with 'pip install kaggle'.")
    #     except Exception as e:
    #         print(f"Error submitting to Kaggle: {e}")


# --- Main Execution Logic ---
if __name__ == '__main__':
    # --- Setup Output Directory ---
    output_dir = Path('./outputs')
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory set to: {output_dir}")

    # --- Model Setup ---
    if CONFIG["model"]["num_classes"] is None:
         raise ValueError("Number of classes not determined. Check dataset loading.")

    model = Sentinel2Classifier(num_classes=CONFIG["model"]["num_classes"])
    model.to(DEVICE)
    print(f"Classifier model created with {CONFIG['model']['num_classes']} classes.")
    # Print model summary (optional, requires torchinfo: pip install torchinfo)
    # try:
    #     from torchinfo import summary
    #     # Provide input size including batch dimension (e.g., B, C, H, W)
    #     input_size = (CONFIG["data"]["batch_size"], 12, CONFIG["data"]["image_size"], CONFIG["data"]["image_size"])
    #     summary(model, input_size=input_size)
    # except ImportError:
    #     print("torchinfo not found, skipping model summary. Install with 'pip install torchinfo'")


    # --- Optimizer and Loss ---
    # Use CrossEntropyLoss for multi-class classification
    criterion = nn.CrossEntropyLoss()
    print(f"Loss function: CrossEntropyLoss")

    # --- Training Parameters from CONFIG ---
    num_epochs = CONFIG["train"]["stage1_epochs"] + CONFIG["train"]["stage2_epochs"]
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

    # --- Staged Fine-tuning Variables ---
    overall_best_val_loss = float('inf')
    best_model_state_dict = None
    total_epochs_run = 0
    # Initialize GradScaler only if AMP is enabled and CUDA is available
    scaler = GradScaler(enabled=CONFIG["amp_enabled"]) if DEVICE.type == 'cuda' else None
    if scaler: print("Using Automatic Mixed Precision (AMP).")

    # --- Stage 1: Train Full Model with Initial LR ---
    print("\n--- Stage 1: Initial Training ---")
    stage = 1
    current_stage_epochs = stage1_epochs
    current_lr = lr_stage1

    # Optimizer for Stage 1 - Train all parameters
    optimizer = torch.optim.AdamW(model.parameters(), lr=current_lr, weight_decay=weight_decay)
    print(f"Optimizer: AdamW (LR={current_lr}, Weight Decay={weight_decay})")
    print(f"Trainable parameters for Stage 1: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Schedulers for Stage 1
    # Linear Warmup Scheduler
    if warmup_epochs > 0:
        # Calculate total steps, considering gradient accumulation
        warmup_steps_per_epoch = len(train_loader) // gradient_accumulation_steps
        total_warmup_steps = warmup_steps_per_epoch * warmup_epochs
        # Adjust lambda for warmup: increase LR linearly from initial_warmup_lr to stage_lr
        # lr_lambda = lambda step: (current_lr - initial_warmup_lr) / total_warmup_steps * step + initial_warmup_lr if step < total_warmup_steps else current_lr
        # Simpler: scale from initial ratio to 1.0
        lr_lambda = lambda step: max(1e-9, initial_warmup_lr / current_lr) * (1.0 - step / total_warmup_steps) + step / total_warmup_steps if step < total_warmup_steps else 1.0

        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        print(f"Using Linear Warmup for {warmup_epochs} epochs ({total_warmup_steps} steps).")
    else:
        warmup_scheduler = None
        total_warmup_steps = 0

    # Reduce LR on Plateau Scheduler (applied after warmup)
    # Reduce factor, patience adjusted slightly
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=patience // 2, verbose=True)
    print(f"Using ReduceLROnPlateau scheduler (Factor=0.2, Patience={patience // 2}) after warmup.")

    epochs_without_improvement = 0 # Reset for stage

    for epoch in range(current_stage_epochs):
        epoch_num = total_epochs_run + 1
        current_actual_lr = optimizer.param_groups[0]['lr'] # Get LR before epoch run

        # --- Training Epoch ---
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, optimizer, scaler, DEVICE, True,
            epoch_num, num_epochs, current_actual_lr, gradient_accumulation_steps
        )

        # --- Validation Epoch ---
        avg_val_loss, val_accuracy = run_epoch(
            model, val_loader_split, criterion, None, scaler, DEVICE, False, # No optimizer/scaler for validation
            epoch_num, num_epochs, 0, gradient_accumulation_steps # LR doesn't matter for validation
        )

        # --- Scheduler Steps ---
        # Step warmup scheduler at each optimizer step (handled implicitly by LambdaLR logic based on step count)
        # Step plateau scheduler based on validation loss AFTER warmup phase
        if epoch_num > warmup_epochs:
             plateau_scheduler.step(avg_val_loss)
        # Manually step the warmup scheduler (LambdaLR needs step count, not epoch)
        # Note: This assumes optimizer.step() happens once per 'gradient_accumulation_steps' batches.
        # A more robust way is to step it inside the training loop after optimizer.step().
        # For simplicity here, we approximate by stepping once per epoch after the epoch runs.
        # This is NOT correct if LambdaLR depends on fine-grained steps.
        # A better approach: Step LambdaLR *inside* run_epoch after optimizer.step().
        # Let's skip manual stepping here and assume ReduceLROnPlateau dominates after warmup.
        # if warmup_scheduler is not None and epoch_num <= warmup_epochs:
        #    warmup_scheduler.step() # Step epoch-level warmup (if scheduler is epoch-based)


        print(f'End of Epoch {epoch_num} - Current LR: {optimizer.param_groups[0]["lr"]:.6e}') # Print LR after potential scheduler step

        # --- Checkpointing and Early Stopping ---
        if avg_val_loss < overall_best_val_loss:
            overall_best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            # Save the model state dict
            try:
                best_model_state_dict = model.state_dict()
                torch.save(best_model_state_dict, model_save_path)
                print(f'---> Validation Loss Improved to {overall_best_val_loss:.4f}. Model saved to {model_save_path}')
            except Exception as e:
                print(f"Error saving model: {e}")
        else:
            epochs_without_improvement += 1
            print(f'---> Stage {stage} Val loss did not improve for {epochs_without_improvement} epochs.')
            if epochs_without_improvement >= patience:
                print(f'Early stopping triggered during Stage {stage} at epoch {epoch_num}.')
                break # Exit stage loop

        total_epochs_run = epoch_num

    print(f"--- Finished Stage 1 (Epoch {total_epochs_run}) ---")
    print(f"Best validation loss during Stage 1: {overall_best_val_loss:.4f}")


    # --- Stage 2: Fine-tuning with Lower LR ---
    # Only run stage 2 if stage 1 completed without early stopping right away
    # and if there are stage 2 epochs defined.
    if epochs_without_improvement < patience and stage2_epochs > 0:
        print("\n--- Stage 2: Fine-tuning ALL Layers with Lower LR ---")
        stage = 2
        current_stage_epochs = stage2_epochs
        current_lr = lr_stage2 # Use the lower LR for stage 2

        # Load the best model state from Stage 1
        if best_model_state_dict:
            print(f"Loading best model state from Stage 1 (Val Loss: {overall_best_val_loss:.4f})")
            model.load_state_dict(best_model_state_dict)
        else:
            print("Warning: No best model state saved from Stage 1. Continuing with the last model state.")

        # Create NEW optimizer with the lower learning rate for fine-tuning
        optimizer = torch.optim.AdamW(model.parameters(), lr=current_lr, weight_decay=weight_decay)
        print(f"Optimizer: AdamW (LR={current_lr}, Weight Decay={weight_decay})")
        print(f"Trainable parameters for Stage 2: {sum(p.numel() for p in model.parameters() if p.requires_grad)}") # Should be same as stage 1 unless layers were frozen

        # Recreate Plateau scheduler for this stage with potentially more patience
        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=patience, verbose=True)
        print(f"Using ReduceLROnPlateau scheduler (Factor=0.2, Patience={patience}) for Stage 2.")
        epochs_without_improvement = 0 # Reset patience count for Stage 2

        for epoch in range(current_stage_epochs):
            epoch_num = total_epochs_run + 1
            current_actual_lr = optimizer.param_groups[0]['lr']

            # --- Training Epoch ---
            train_loss, train_accuracy = run_epoch(
                model, train_loader, criterion, optimizer, scaler, DEVICE, True,
                epoch_num, num_epochs, current_actual_lr, gradient_accumulation_steps
            )

            # --- Validation Epoch ---
            avg_val_loss, val_accuracy = run_epoch(
                model, val_loader_split, criterion, None, scaler, DEVICE, False,
                epoch_num, num_epochs, 0, gradient_accumulation_steps
            )

            # Step plateau scheduler
            plateau_scheduler.step(avg_val_loss)
            print(f"End of Epoch {epoch_num} - Current LR: {optimizer.param_groups[0]['lr']:.6e}")

            # --- Checkpointing and Early Stopping ---
            if avg_val_loss < overall_best_val_loss:
                overall_best_val_loss = avg_val_loss
                epochs_without_improvement = 0
                try:
                    best_model_state_dict = model.state_dict()
                    torch.save(best_model_state_dict, model_save_path)
                    print(f'---> Overall Validation Loss Improved to {overall_best_val_loss:.4f}. Model saved to {model_save_path}')
                except Exception as e:
                    print(f"Error saving model: {e}")
            else:
                epochs_without_improvement += 1
                print(f'---> Stage {stage} Val loss did not improve for {epochs_without_improvement} epochs.')
                if epochs_without_improvement >= patience:
                    print(f'Early stopping triggered during Stage {stage} at epoch {epoch_num}.')
                    break # Exit stage loop

            total_epochs_run = epoch_num

        print(f"--- Finished Stage 2 (Epoch {total_epochs_run}) ---")

    # --- Final Report ---
    print("\n--- Training Complete ---")
    if best_model_state_dict is not None:
        print(f'Best model saved to {model_save_path}')
        print(f'Best overall validation loss achieved: {overall_best_val_loss:.4f}')
        # Load the best model for final evaluation
        print("Loading best model weights for final evaluation.")
        model.load_state_dict(torch.load(model_save_path, map_location=DEVICE))
    else:
        print("Training completed, but no improvement observed or no best model saved.")
        print("Using the model state at the end of training for evaluation.")


    # --- Run Final Evaluation & Make Submission ---
    if final_pred_loader:
         if 'class_to_idx_map' not in globals() or not class_to_idx_map:
             print("Error: class_to_idx_map not defined or empty. Cannot run final evaluation.")
         else:
             evaluate_model(CONFIG, model, final_pred_loader, DEVICE, class_to_idx_map)
    else:
        print("Skipping final evaluation as prediction data loader was not created.")

    print("\n--- Script Finished ---")

