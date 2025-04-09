import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
# Use torch.amp directly for GradScaler and autocast
from torch.amp import GradScaler, autocast

import rasterio
import numpy as np
import pandas as pd
import os
import random
import time
import traceback
from pathlib import Path
from sklearn.model_selection import train_test_split
# Use standard tqdm, notebook version might cause issues in non-notebook environments
from tqdm import tqdm
from typing import Tuple, List, Dict, Any, Optional, Callable

# --- Configuration ---
CONFIG = {
    "seed": 1337,
    "data": {
        "train_dir": Path('./ds/images/remote_sensing/otherDatasets/sentinel_2/tif'),
        "prediction_dir": Path('./testset/testset'), # Directory with .npy files for final prediction
        "output_dir": Path('./outputs'),
        "train_ratio": 0.9,
        "num_workers": 4,
    },
    "model": {
        "name": "Sentinel2Classifier_v2",
        "save_path": Path('./outputs/sentinel2_classifier_v2_per_image_norm.pth'),
        "num_classes": None, # Determined dynamically from data
    },
    "training": {
        # Determine device, checking MPS availability specifically
        "device": "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cpu",
        "image_size": 128,
        "batch_size": 16,
        "gradient_accumulation_steps": 2, # Accumulate gradients over 2 steps
        "optimizer": "AdamW",
        "weight_decay": 3.7553e-05,
        "criterion": nn.CrossEntropyLoss,
        "amp_enabled": True, # Enable Automatic Mixed Precision (only effective on CUDA)
        "staged_training": True,
        "stage1": {
            "epochs": 30,
            "lr": 1e-3,
            "freeze_backbone": False, # Set True to only train head/projector initially
            "warmup_epochs": 3,
            "initial_warmup_lr": 1e-6,
            "patience": 5, # Early stopping patience for this stage
        },
        "stage2": {
            "epochs": 30,
            "lr": 1e-4, # Lower LR for full fine-tuning
            "patience": 10, # Early stopping patience for this stage
        },
    },
    "prediction": {
        "predictions_csv_path": Path('./outputs/submission_track2.csv'),
        "kaggle_competition": 'aicrowd-geospatial-challenge', # Replace with actual competition slug if needed
        "kaggle_message": 'Submission with Sentinel2Classifier v2',
    }
}

# --- Setup ---
CONFIG["data"]["output_dir"].mkdir(parents=True, exist_ok=True)
DEVICE = torch.device(CONFIG["training"]["device"])
# Determine if AMP is *actually* usable (requires CUDA)
AMP_ENABLED = CONFIG["training"]["amp_enabled"] and DEVICE.type == 'cuda'

print(f"Using device: {DEVICE}")
print(f"Automatic Mixed Precision (AMP) enabled: {AMP_ENABLED}")


# --- Seed Everything ---
def seed_everything(seed: int):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Keep deterministic settings if possible, but benchmark=False can slow down training
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # Set False for full determinism

seed_everything(CONFIG["seed"])

# --- Data Loading Functions ---
def load_sentinel2_image(filepath: Path) -> np.ndarray:
    """Loads a Sentinel-2 image (TIF or NPY), returns NumPy CHW (12 bands)."""
    filepath_str = str(filepath) # rasterio needs string paths

    if filepath.suffix.lower() in ['.tif', '.tiff']:
        try:
            with rasterio.open(filepath_str) as src:
                # Bands 1-9, 11-13 (skip B10 - thermal, B1 - coastal aerosol)
                # Adjust band indices if needed based on specific dataset requirements
                bands_to_read = list(range(1, 10)) + list(range(11, 14)) # 12 bands

                if src.count < max(bands_to_read):
                     raise ValueError(f"Expected at least {max(bands_to_read)} bands, got {src.count} in {filepath}")
                image = src.read(bands_to_read) # Shape: (12, H, W)
        except rasterio.RasterioIOError as e:
            raise IOError(f"Error reading TIF file {filepath}: {e}")

    elif filepath.suffix.lower() == '.npy':
        try:
            image = np.load(filepath_str)
            # Assuming NPY files are already processed and have 12 channels
            if image.ndim != 3 or image.shape[0] != 12:
                 raise ValueError(f"Unexpected shape for .npy {filepath}: {image.shape}. Expected (12, H, W).")
        except Exception as e:
            raise IOError(f"Error loading NPY file {filepath}: {e}")

    else:
        raise ValueError(f"Unsupported file type: {filepath}")

    return image.astype(np.float32)

# --- Per-Image Normalization Function ---
def normalize_image_per_image(image_np: np.ndarray) -> np.ndarray:
    """Normalizes 12-channel NumPy image (C, H, W) using its own stats."""
    if image_np.ndim != 3 or image_np.shape[0] != 12:
        raise ValueError(f"Invalid shape for per-image normalization: {image_np.shape}. Expected (12, H, W).")

    mean = np.mean(image_np, axis=(1, 2), keepdims=True)
    std = np.std(image_np, axis=(1, 2), keepdims=True)
    std[std == 0] = 1e-7 # Avoid division by zero

    return (image_np - mean) / std

# --- Dataset Classes ---
class Sentinel2Dataset(Dataset):
    """Dataset for Sentinel-2 TIF images with labels."""
    def __init__(self, paths_labels: List[Tuple[Path, int]], transform: Optional[Callable] = None):
        self.paths_labels = paths_labels
        self.transform = transform
        if not self.paths_labels:
            print("Warning: Sentinel2Dataset initialized with empty paths_labels list.")

    def __len__(self) -> int:
        return len(self.paths_labels)

    def __getitem__(self, idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor, str]]:
        image_path, label = self.paths_labels[idx]
        try:
            image_np = load_sentinel2_image(image_path)
            image_np = normalize_image_per_image(image_np) # Apply per-image normalization

            image_tensor = torch.from_numpy(image_np.copy()).float() # Convert to tensor
            label_tensor = torch.tensor(label, dtype=torch.long)

            if self.transform:
                image_tensor = self.transform(image_tensor) # Apply augmentations/resizing

            return image_tensor, label_tensor, str(image_path)

        except Exception as e:
            print(f"ERROR loading/processing image {image_path}: {e}\n{traceback.format_exc()}")
            # Returning None signals an error to the custom collate function
            return None

class NpyPredictionDataset(Dataset):
    """Dataset for Sentinel-2 NPY images for prediction (no labels)."""
    def __init__(self, root_dir: Path, transform: Optional[Callable] = None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_paths = sorted([p for p in root_dir.glob('*.npy')])
        if not self.file_paths:
            raise FileNotFoundError(f"No .npy files found in {root_dir}")
        print(f"Found {len(self.file_paths)} .npy files for prediction in {root_dir}.")

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Optional[Tuple[torch.Tensor, str]]:
        image_path = self.file_paths[idx]
        try:
            image_np = load_sentinel2_image(image_path)
            image_np = normalize_image_per_image(image_np) # Apply per-image normalization
            image_tensor = torch.from_numpy(image_np.copy()).float() # Convert to tensor

            if self.transform:
                image_tensor = self.transform(image_tensor) # Apply resizing

            # Return image tensor and its original path
            return image_tensor, str(image_path)

        except Exception as e:
            print(f"ERROR loading/processing prediction image {image_path}: {e}\n{traceback.format_exc()}")
            return None

# --- Custom Collate Function ---
def safe_collate(batch: List[Optional[Any]]) -> Optional[List[Any]]:
    """Collate function that filters out None results from failed __getitem__ calls."""
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None # Return None if the whole batch failed
    # Default collate can handle the rest if the batch is not empty
    return torch.utils.data.dataloader.default_collate(batch)


# --- Data Transforms ---
class AddGaussianNoise:
    """Adds Gaussian noise to a tensor."""
    def __init__(self, mean: float = 0., std: float = 0.1):
        self.std = std
        self.mean = mean

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor + torch.randn_like(tensor) * self.std + self.mean

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"

class RandomChannelScale:
    """Applies random scaling to each channel independently."""
    def __init__(self, scale_range: Tuple[float, float] = (0.9, 1.1)):
        self.scale_range = scale_range

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        # Generate scale factors on the same device as the tensor
        scale_factors = torch.rand(tensor.shape[0], device=tensor.device) * (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0]
        return tensor * scale_factors.view(tensor.shape[0], 1, 1)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(scale_range={self.scale_range})"

# Define transforms based on config
img_size = CONFIG["training"]["image_size"]

train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    # Consider smaller rotation/affine ranges if they distort Sentinel data too much
    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    # ColorJitter might not be suitable for non-RGB satellite data, comment out unless validated
    # transforms.ColorJitter(brightness=0.1, contrast=0.1),
    AddGaussianNoise(mean=0., std=0.03),
    RandomChannelScale(scale_range=(0.95, 1.05)),
    transforms.Resize((img_size, img_size), antialias=True),
    # Normalization is done per-image in the Dataset __getitem__
])

val_pred_transforms = transforms.Compose([
    transforms.Resize((img_size, img_size), antialias=True),
    # Normalization is done per-image in the Dataset __getitem__
])


# --- Model Architecture ---
class Mish(nn.Module):
    """Mish Activation Function: f(x) = x * tanh(softplus(x))"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(F.softplus(x))

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            Mish(),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.excitation(self.squeeze(x))
        return x * scale

class BasicConvBlock(nn.Module):
    """Basic Convolutional Block: Conv -> BN -> Activation -> SE (optional)."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1, use_se: bool = True):
        super().__init__()
        self.use_se = use_se
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False) # Bias False with BN
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = Mish()
        if self.use_se:
            self.se = SEBlock(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        if self.use_se:
            x = self.se(x)
        return x

class ResBlock(nn.Module):
    """Improved Residual Block."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = BasicConvBlock(channels, channels, use_se=False) # SE only after final add
        self.conv2 = BasicConvBlock(channels, channels, use_se=False)
        self.se = SEBlock(channels)
        # No 1x1 conv needed here as input/output channels are the same

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.se(out + residual) # Add residual *before* final SE block
        return out

class Sentinel2Classifier_v2(nn.Module):
    """Revised CNN with standard ResBlocks, Attention, Hybrid Pooling & Mish."""
    def __init__(self, num_classes: int, in_channels: int = 12):
        super().__init__()

        # Initial convolution to increase channels
        self.stem = BasicConvBlock(in_channels, 64, use_se=True) # 12 -> 64

        # Downsampling/Feature Extraction Stages with ResBlocks
        self.layer1 = self._make_layer(64, 128, num_blocks=2) # 64 -> 128
        self.layer2 = self._make_layer(128, 256, num_blocks=2) # 128 -> 256
        self.layer3 = self._make_layer(256, 512, num_blocks=2) # 256 -> 512
        # self.layer4 = self._make_layer(512, 1024, num_blocks=1) # 512 -> 1024 (Optional deeper layer)

        # Pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Classifier Head
        # feature_dim = 1024 if hasattr(self, 'layer4') else 512
        feature_dim = 512 # Based on layer3 output
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(feature_dim * 2, 256), # *2 for concat avg+max pool
            Mish(),
            nn.BatchNorm1d(256), # BN after activation, before dropout
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

        self.apply(self._initialize_weights)
        print(f"Initialized {self.__class__.__name__} with {feature_dim*2} features -> {num_classes} classes.")

    def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int) -> nn.Sequential:
        """Creates a layer consisting of a downsampling block and ResBlocks."""
        layers = []
        # Downsampling block (Conv stride 2) + SE
        layers.append(BasicConvBlock(in_channels, out_channels, stride=2, use_se=True))
        # Residual blocks
        for _ in range(num_blocks):
            layers.append(ResBlock(out_channels))
        return nn.Sequential(*layers)

    def _initialize_weights(self, m: nn.Module):
        """Initialize weights using Kaiming normal for Conv, Xavier for Linear."""
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') # Mish is relu-like
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # if hasattr(self, 'layer4'): x = self.layer4(x)

        # Hybrid Pooling
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        x = torch.cat([avg_out, max_out], dim=1) # Concatenate features

        x = self.head(x)
        return x

# --- Training & Validation Epoch ---
def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[optim.Optimizer], # Optional for validation
    scaler: GradScaler, # Always pass scaler, its 'enabled' state controls behavior
    device: torch.device,
    is_training: bool,
    epoch_num: int,
    num_epochs_total: int,
    current_lr: float,
    warmup_scheduler: Optional[LambdaLR] = None,
    grad_accum_steps: int = 1
) -> Tuple[float, float]:
    """Runs a single epoch of training or validation."""
    if is_training:
        model.train()
        if optimizer is None: raise ValueError("Optimizer must be provided for training.")
    else:
        model.eval()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    start_time = time.time()

    # Determine context based on training/validation
    context = torch.enable_grad() if is_training else torch.no_grad()
    # Use keyword arguments for autocast and rely on scaler's enabled state
    # scaler.is_enabled() returns True only if AMP is enabled in config AND device is CUDA
    amp_context = autocast(device_type=device.type, enabled=scaler.is_enabled())

    loader_desc = "Train" if is_training else "Valid"
    progress_bar = tqdm(loader, desc=f"Epoch {epoch_num}/{num_epochs_total} ({loader_desc}) LR={current_lr:.2e}", leave=False)

    if is_training: optimizer.zero_grad() # Zero gradients at the beginning of the epoch

    with context:
        for batch_idx, batch_data in enumerate(progress_bar):
            if batch_data is None: # Handle failed batch from collate_fn
                 print(f"Warning: Skipping None batch at index {batch_idx}")
                 continue

            # Adapt based on dataset type (train/val vs pred)
            if is_training or isinstance(loader.dataset, Sentinel2Dataset):
                 inputs, targets, _ = batch_data
                 targets = targets.to(device)
            else: # Prediction dataset
                 inputs, _ = batch_data # No targets

            inputs = inputs.to(device)

            # Use autocast context manager
            with amp_context:
                outputs = model(inputs)
                if is_training or isinstance(loader.dataset, Sentinel2Dataset):
                    loss = criterion(outputs, targets)
                    # Scale loss for gradient accumulation ONLY if training
                    if is_training:
                        scaled_loss = loss / grad_accum_steps
                else:
                    loss = torch.tensor(0.0) # No loss calculation for prediction

            # --- Backpropagation and Optimization ---
            if is_training:
                # scaler.scale() first, then scaler.step() and scaler.update()
                scaler.scale(scaled_loss).backward() # scaler handles enabled state

                # Gradient accumulation step
                if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(loader):
                    scaler.step(optimizer) # scaler handles enabled state
                    scaler.update() # scaler handles enabled state
                    optimizer.zero_grad(set_to_none=True) # More memory efficient

                    # Step iteration-level scheduler (warmup) AFTER optimizer step
                    if warmup_scheduler:
                        warmup_scheduler.step()

            # --- Metrics Calculation ---
            if is_training or isinstance(loader.dataset, Sentinel2Dataset):
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_samples += targets.size(0)
                correct_predictions += (predicted == targets).sum().item()
                # Update progress bar
                current_acc = 100. * correct_predictions / total_samples if total_samples > 0 else 0
                progress_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{current_acc:.2f}%")
            else:
                 total_samples += inputs.size(0) # Count samples processed for prediction

    epoch_duration = time.time() - start_time
    avg_loss = running_loss / total_samples if total_samples > 0 else 0
    accuracy = 100. * correct_predictions / total_samples if total_samples > 0 else 0

    if is_training or isinstance(loader.dataset, Sentinel2Dataset):
        print(f'Epoch [{epoch_num}/{num_epochs_total}] FINISHED ({loader_desc}) - Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%, Time: {epoch_duration:.2f}s')
    else:
        print(f'Epoch [{epoch_num}/{num_epochs_total}] FINISHED ({loader_desc}) - Samples Processed: {total_samples}, Time: {epoch_duration:.2f}s')

    return avg_loss, accuracy


# --- Data Loading Setup ---
def create_dataloaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataLoader, int, List[str]]:
    """Creates datasets and dataloaders based on the configuration."""
    print("Scanning training directory and creating splits...")
    train_dir = config["data"]["train_dir"]
    train_ratio = config["data"]["train_ratio"]
    seed = config["seed"]
    batch_size = config["training"]["batch_size"]
    num_workers = config["data"]["num_workers"]
    pred_dir = config["data"]["prediction_dir"]

    full_dataset_paths_labels: List[Tuple[Path, int]] = []
    class_to_idx_map: Dict[str, int] = {}
    class_names: List[str] = []

    # Scan training directory
    for class_folder in sorted(train_dir.iterdir()):
        if class_folder.is_dir() and not class_folder.name.startswith('.'):
            class_name = class_folder.name
            if class_name not in class_to_idx_map:
                class_idx = len(class_names)
                class_to_idx_map[class_name] = class_idx
                class_names.append(class_name)
            else:
                class_idx = class_to_idx_map[class_name]

            for filepath in class_folder.glob('*.tif'):
                full_dataset_paths_labels.append((filepath, class_idx))
            for filepath in class_folder.glob('*.tiff'): # Also check for .tiff
                full_dataset_paths_labels.append((filepath, class_idx))


    num_classes = len(class_names)
    if num_classes == 0:
        raise FileNotFoundError(f"No valid class folders found in {train_dir}")
    print(f"Found {len(full_dataset_paths_labels)} training images across {num_classes} classes: {class_names}")

    # Stratified Split
    labels = [label for _, label in full_dataset_paths_labels]
    if len(set(labels)) < 2:
         print("Warning: Only one class found. Cannot perform stratified split. Using random split.")
         train_info, val_info = train_test_split(
             full_dataset_paths_labels, train_size=train_ratio, random_state=seed
         )
    elif len(full_dataset_paths_labels) > 1 :
         try:
             train_info, val_info = train_test_split(
                 full_dataset_paths_labels, train_size=train_ratio, random_state=seed,
                 stratify=labels
             )
         except ValueError as e:
             print(f"Stratified split failed ({e}). Falling back to random split.")
             train_info, val_info = train_test_split(
                 full_dataset_paths_labels, train_size=train_ratio, random_state=seed
             )
    else:
        # Handle case with 0 or 1 sample
        train_info = full_dataset_paths_labels
        val_info = []
        print("Warning: Very few samples, using all for training, none for validation split.")


    print(f"Split: {len(train_info)} train, {len(val_info)} validation samples.")

    # Create Dataset objects
    train_dataset = Sentinel2Dataset(train_info, transform=train_transforms)
    val_dataset = Sentinel2Dataset(val_info, transform=val_pred_transforms)
    # Handle case where prediction directory might not exist or be needed initially
    try:
        pred_dataset = NpyPredictionDataset(pred_dir, transform=val_pred_transforms)
        pred_loader = DataLoader(
            pred_dataset, batch_size=batch_size * 2, shuffle=False,
            num_workers=num_workers, pin_memory=True, collate_fn=safe_collate # Use safe collate
        )
    except FileNotFoundError:
        print(f"Warning: Prediction directory {pred_dir} not found. Prediction loader not created.")
        pred_loader = None


    # Create DataLoaders
    # Use a generator for reproducibility in DataLoader shuffling
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, generator=g, collate_fn=safe_collate # Use safe collate
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size * 2, # Often use larger batch for validation
        pin_memory=True, shuffle=False, num_workers=num_workers, collate_fn=safe_collate
    )

    print("DataLoaders created.")
    return train_loader, val_loader, pred_loader, num_classes, class_names


# --- Prediction and Submission ---
def predict_and_submit(model: nn.Module, pred_loader: Optional[DataLoader], config: Dict[str, Any], device: torch.device, class_names: List[str]):
    """Runs prediction on the test set and generates a submission file."""
    if pred_loader is None:
        print("Prediction loader not available. Skipping prediction and submission.")
        return

    print("\n--- Starting Prediction Phase ---")
    model.eval()
    predictions = []
    image_ids = [] # Store corresponding image IDs (filenames)

    # Load best model state
    model_path = config["model"]["save_path"]
    if model_path.exists():
        print(f"Loading best model weights from: {model_path}")
        # Load state dict onto the correct device
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Warning: Model file not found at {model_path}. Using current model state for prediction.")

    # Recreate scaler for prediction phase, respecting AMP_ENABLED
    # Remove device_type argument
    amp_enabled_pred = config["training"]["amp_enabled"] and device.type == 'cuda'
    scaler = GradScaler(enabled=amp_enabled_pred)

    with torch.no_grad():
        progress_bar = tqdm(pred_loader, desc="Predicting")
        for batch_data in progress_bar:
            if batch_data is None: continue # Skip failed batches

            inputs, paths = batch_data # Unpack image tensors and paths
            inputs = inputs.to(device)

            # Use keyword arguments for autocast
            with autocast(device_type=device.type, enabled=scaler.is_enabled()):
                outputs = model(inputs)

            # Get predicted class indices
            _, predicted_indices = torch.max(outputs, 1)
            predictions.extend(predicted_indices.cpu().numpy())

            # Extract image IDs (e.g., filename without extension)
            batch_ids = [Path(p).stem for p in paths] # Get 'test_0000' etc.
            image_ids.extend(batch_ids)

    if not predictions:
        print("Error: No predictions were generated. Check the prediction dataset and loader.")
        return

    # Create DataFrame
    print(f"Generated {len(predictions)} predictions for {len(image_ids)} images.")
    if len(predictions) != len(image_ids):
         print(f"Warning: Mismatch between number of predictions ({len(predictions)}) and image IDs ({len(image_ids)}).")
         # Attempt to align if possible, otherwise report error
         min_len = min(len(predictions), len(image_ids))
         predictions = predictions[:min_len]
         image_ids = image_ids[:min_len]
         print(f"Using {min_len} aligned predictions/IDs.")


    # Map predicted indices to class names
    predicted_class_names = [class_names[idx] for idx in predictions]

    pred_df = pd.DataFrame({
        'test_id': image_ids,
        'predicted_class': predicted_class_names # Use class names directly
    })

    # Ensure test_id is just the number/identifier part if needed by competition format
    # Example: if image_ids are 'test_0001', extract '0001'
    # pred_df['test_id'] = pred_df['test_id'].str.replace('test_', '', regex=False) # Uncomment if needed

    # Save predictions
    csv_path = config["prediction"]["predictions_csv_path"]
    pred_df.to_csv(csv_path, index=False)
    print(f"Predictions saved to: {csv_path}")

    # --- Kaggle Submission (Optional) ---
    submit_to_kaggle = True # Set to False to skip submission
    if submit_to_kaggle:
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate() # Ensure kaggle.json is in ~/.kaggle/ or C:\Users\<User>\.kaggle

            competition = config["prediction"]["kaggle_competition"]
            message = config["prediction"]["kaggle_message"]

            print(f"Attempting submission to Kaggle competition: {competition}")
            api.competition_submit(
                 file_name=str(csv_path),
                 message=message,
                 competition=competition
            )
            print("Submission successful!")
        except ImportError:
            print("Kaggle API not found. Skipping submission. Install with: pip install kaggle")
        except Exception as e:
            print(f"Kaggle API submission failed: {e}")
            print("Please ensure your kaggle.json is configured correctly and the competition slug is accurate.")


# --- Main Execution Logic ---
if __name__ == '__main__':

    # --- Data ---
    train_loader, val_loader, pred_loader, num_classes, class_names = create_dataloaders(CONFIG)
    CONFIG["model"]["num_classes"] = num_classes # Update config with dynamic num_classes

    # --- Model ---
    model = Sentinel2Classifier_v2(num_classes=num_classes)
    model.to(DEVICE)

    # --- Loss Function ---
    criterion = CONFIG["training"]["criterion"]()

    # --- Training Variables ---
    overall_best_val_loss = float('inf')
    best_model_state_dict = None
    total_epochs_run = 0
    # Use the global AMP_ENABLED flag determined earlier
    # Remove device_type argument from GradScaler constructor
    scaler = GradScaler(enabled=AMP_ENABLED)

    # --- Training Loop ---
    stages = []
    if CONFIG["training"]["staged_training"]:
        stages.append(("Stage 1", CONFIG["training"]["stage1"]))
        stages.append(("Stage 2", CONFIG["training"]["stage2"]))
    else:
        # Combine epochs if not staged
        single_stage_config = CONFIG["training"]["stage1"].copy()
        single_stage_config["epochs"] = CONFIG["training"]["stage1"]["epochs"] + CONFIG["training"]["stage2"]["epochs"]
        single_stage_config["patience"] = CONFIG["training"]["stage2"]["patience"] # Use longer patience
        single_stage_config["freeze_backbone"] = False # Ensure all layers train
        stages.append(("Single Stage", single_stage_config))


    for stage_name, stage_config in stages:
        print(f"\n--- {stage_name}: {stage_config['epochs']} Epochs ---")
        stage_epochs = stage_config["epochs"]
        stage_lr = stage_config["lr"]
        stage_patience = stage_config["patience"]
        freeze_backbone = stage_config.get("freeze_backbone", False) # Default to False

        # --- Optimizer and Schedulers for the current stage ---
        # Handle potential freezing (Example: freeze all except head for stage 1)
        if freeze_backbone:
             print("Freezing backbone, training only the head.")
             for name, param in model.named_parameters():
                 if not name.startswith('head.'):
                     param.requires_grad = False
             # Filter parameters for the optimizer
             optimizer_params = filter(lambda p: p.requires_grad, model.parameters())
             # Convert filter object to list for AdamW/Adam
             optimizer_params = list(optimizer_params)
             if not optimizer_params:
                 print("Warning: No parameters to train in head. Check model structure or freeze logic.")
                 continue # Skip this stage if nothing to train
        else:
             print("Training all parameters.")
             for param in model.parameters(): # Ensure all are trainable if not freezing
                  param.requires_grad = True
             optimizer_params = model.parameters()

        # Count trainable parameters for verification
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters for {stage_name}: {trainable_params}")

        # Define optimizer
        if CONFIG["training"]["optimizer"].lower() == 'adamw':
            optimizer = optim.AdamW(optimizer_params, lr=stage_lr, weight_decay=CONFIG["training"]["weight_decay"])
        else: # Default or other optimizers like Adam
            optimizer = optim.Adam(optimizer_params, lr=stage_lr, weight_decay=CONFIG["training"]["weight_decay"])

        # Warmup scheduler (only if configured for this stage)
        warmup_epochs = stage_config.get("warmup_epochs", 0)
        warmup_scheduler = None
        if warmup_epochs > 0 and stage_name == "Stage 1": # Typically only warmup in the first stage
            initial_warmup_lr = stage_config["initial_warmup_lr"]
            # Calculate steps considering gradient accumulation
            warmup_steps_per_epoch = len(train_loader) // CONFIG["training"]["gradient_accumulation_steps"]
            total_warmup_steps = warmup_steps_per_epoch * warmup_epochs
            if total_warmup_steps == 0: total_warmup_steps = 1 # Avoid division by zero if loader is small

            # Linear warmup from initial_lr to stage_lr
            def lr_lambda(step):
                if step < total_warmup_steps:
                    # Calculate progress from 0 to 1 during warmup
                    progress = float(step) / float(total_warmup_steps)
                    # Interpolate linearly between initial and target LR ratio
                    lr_ratio = (initial_warmup_lr / stage_lr) * (1.0 - progress) + progress
                    return lr_ratio
                else:
                    # After warmup, maintain the target LR ratio (1.0)
                    return 1.0

            warmup_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
            print(f"Using linear warmup for {warmup_epochs} epochs ({total_warmup_steps} steps) in {stage_name}.")

        # Plateau scheduler for reducing LR after warmup/initial phase
        # Factor=0.1 means LR reduces by 10x. Removed verbose=True.
        plateau_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=stage_patience // 2)

        epochs_without_improvement = 0

        # --- Epoch Loop for the current stage ---
        for epoch in range(stage_epochs):
            current_epoch_num = total_epochs_run + 1
            current_lr = optimizer.param_groups[0]['lr'] # Get LR before epoch run

            # Run Training Epoch
            run_epoch(
                model, train_loader, criterion, optimizer, scaler, DEVICE, True,
                current_epoch_num, CONFIG["training"]["stage1"]["epochs"] + CONFIG["training"]["stage2"]["epochs"], current_lr,
                warmup_scheduler if current_epoch_num <= warmup_epochs else None, # Pass warmup only during its phase
                CONFIG["training"]["gradient_accumulation_steps"]
            )

            # Run Validation Epoch
            avg_val_loss, val_accuracy = run_epoch(
                model, val_loader, criterion, None, scaler, DEVICE, False,
                current_epoch_num, CONFIG["training"]["stage1"]["epochs"] + CONFIG["training"]["stage2"]["epochs"], current_lr
            )

            # Step plateau scheduler AFTER validation and AFTER potential warmup phase
            if current_epoch_num > warmup_epochs:
                 plateau_scheduler.step(avg_val_loss)
            # Get LR *after* potential scheduler steps
            lr_after_step = optimizer.param_groups[0]['lr']
            print(f'End of Epoch {current_epoch_num} - Val Loss: {avg_val_loss:.4f} Val Acc: {val_accuracy:.2f}% - Current LR: {lr_after_step:.4e}')


            # Check for improvement and save best model
            if avg_val_loss < overall_best_val_loss:
                overall_best_val_loss = avg_val_loss
                epochs_without_improvement = 0
                best_model_state_dict = model.state_dict()
                torch.save(best_model_state_dict, CONFIG["model"]["save_path"])
                print(f'---> Overall Validation Loss Improved to {overall_best_val_loss:.4f}. Model saved to {CONFIG["model"]["save_path"]}')
            else:
                epochs_without_improvement += 1
                print(f'---> {stage_name} Val loss did not improve for {epochs_without_improvement} epochs.')
                if epochs_without_improvement >= stage_patience:
                    print(f'Early stopping triggered during {stage_name} at epoch {current_epoch_num}.')
                    break # Stop training this stage

            total_epochs_run = current_epoch_num # Increment total epochs completed

        print(f"--- Finished {stage_name} (Epoch {total_epochs_run}) ---")
        # Break outer loop if early stopping happened in the current stage
        if epochs_without_improvement >= stage_patience:
             break


    # --- Final Report ---
    if best_model_state_dict is not None:
        print(f'\nTraining finished. Best model saved to {CONFIG["model"]["save_path"]} with validation loss: {overall_best_val_loss:.4f}')
        # --- Run Prediction and Submission using the best model ---
        predict_and_submit(model, pred_loader, CONFIG, DEVICE, class_names)
    else:
        print("\nTraining completed, but no improvement observed based on initial validation loss.")
        print("Consider running prediction with the final model state if needed, but it might not be optimal.")
        # Optionally run prediction even if no improvement:
        # predict_and_submit(model, pred_loader, CONFIG, DEVICE, class_names)

    print("Script finished.")
