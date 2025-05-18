import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models
import rasterio
import numpy as np
import torch.nn.functional as F
import pandas as pd
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import random
import time
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import train_test_split
import traceback
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, List, Dict, Any, Optional, Callable
import warnings
# SWA Imports
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR
# YAML and Argparse Imports
import yaml
import argparse

# --- Plotting Imports (NEW) ---
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix as sk_confusion_matrix, classification_report

# --- Argument Parser ---
parser = argparse.ArgumentParser(description='Train ResNet model for EuroSAT classification.')
parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file.')
# Use parse_known_args to avoid issues in some environments (like notebooks)
args, unknown = parser.parse_known_args()


# --- Load Configuration ---
config_path = Path(args.config)
if not config_path.is_file():
    raise FileNotFoundError(f"Configuration file not found: {config_path}")
try:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f) # Load config from YAML
    print(f"Loaded configuration from: {config_path}")
except yaml.YAMLError as e:
    raise ValueError(f"Error parsing YAML file: {e}")
except Exception as e:
    raise IOError(f"Error reading config file: {e}")

# --- Device Setup (Override config if needed) ---
USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    DEVICE = torch.device("cuda")
    config['device'] = 'cuda' # Ensure config reflects actual device
    config['amp_enabled'] = config.get('amp_enabled', True) # Keep True if not specified
elif torch.backends.mps.is_available(): # MPS check
    DEVICE = torch.device("mps")
    config['device'] = 'mps'
    config['amp_enabled'] = False # AMP typically not used/stable on MPS
else:
    DEVICE = torch.device("cpu")
    config['device'] = 'cpu'
    config['amp_enabled'] = False
print(f"Using device: {DEVICE}")
print(f"AMP Enabled: {config['amp_enabled']}")


# --- Suppress warnings ---
warnings.filterwarnings("ignore")

# --- Seed Setting ---
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to {seed}")
# Cast seed to int for robustness
set_seed(int(config["train"]["seed"]))

# --- Data Loading and Preprocessing (Indices BEFORE Norm) ---
TARGET_BANDS_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12] # Corresponds to B1..B8,B8A,B11,B12 (B9 and B10 omitted)

# --- Band Mapping (Standard Order) ---
# This map reflects the standard L2A product band order (12 bands of interest)
# B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B11, B12
BAND_MAP_12 = {
    "B1_Coastal": 0, "B2_Blue": 1, "B3_Green": 2, "B4_Red": 3,
    "B5_RE1": 4, "B6_RE2": 5, "B7_RE3": 6, "B8_NIR": 7,
    "B8A_NIR2": 8, "B9_WV": 9, "B11_SWIR1": 10, "B12_SWIR2": 11
}
print(f"Using BAND_MAP_12 reflecting standard order: {BAND_MAP_12}")

# --- load_sentinel2_image function (with reordering logic) ---
def load_sentinel2_image(filepath: str) -> Optional[np.ndarray]:
    """Loads 13-band TIF, selects 12, reorders to standard L2A sequence."""
    try:
        filepath_lower = filepath.lower()
        if not filepath_lower.endswith(('.tif', '.tiff')): return None
        with rasterio.open(filepath) as src:
            if src.count != 13: return None # Skip if not 13 bands
            all_bands = src.read(list(range(1, 14))).astype(np.float32) # Read all 13 bands

            # Indices to pick from the 13-band TIF to get the standard 12-band L2A order
            # TIF bands: B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11, B12, B8A
            # Desired L2A order (12 bands): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B11, B12
            # Mapping from TIF band index (0-12) to desired position:
            # B1 (TIF 0) -> L2A 0
            # B2 (TIF 1) -> L2A 1
            # ...
            # B8 (TIF 7) -> L2A 7
            # B8A (TIF 12) -> L2A 8  <-- Reordering happens here
            # B9 (TIF 8) -> L2A 9   <-- And here
            # B11 (TIF 10) -> L2A 10
            # B12 (TIF 11) -> L2A 11
            # B10 (TIF 9) is omitted.
            standard_order_indices_in_tif = [0, 1, 2, 3, 4, 5, 6, 7, 12, 8, 10, 11] # B8A is TIF band 12, B9 is TIF band 8
            image_data_12_standard_order = all_bands[standard_order_indices_in_tif, :, :]

            if image_data_12_standard_order.shape[0] != 12: return None
            return image_data_12_standard_order
    except Exception as e: print(f"Error loading/reordering TIF {filepath}: {e}"); return None

# --- Index Calculation ---
def calculate_indices_from_raw(image_np_12bands: np.ndarray, epsilon=1e-7) -> Dict[str, np.ndarray]:
    indices = {}; clip_val = 1.0; global BAND_MAP_12 # Use the global standard band map
    try:
        # Access bands using the standard BAND_MAP_12 keys
        nir=image_np_12bands[BAND_MAP_12["B8_NIR"],:,:]; red=image_np_12bands[BAND_MAP_12["B4_Red"],:,:]
        green=image_np_12bands[BAND_MAP_12["B3_Green"],:,:]; swir1=image_np_12bands[BAND_MAP_12["B11_SWIR1"],:,:]
        re1=image_np_12bands[BAND_MAP_12["B5_RE1"],:,:] # Red Edge 1

        denominator_ndvi=nir+red; ndvi=np.full_like(denominator_ndvi, 0.0, dtype=np.float32)
        valid_mask_ndvi=np.abs(denominator_ndvi)>epsilon; ndvi[valid_mask_ndvi]=(nir[valid_mask_ndvi]-red[valid_mask_ndvi])/denominator_ndvi[valid_mask_ndvi]
        indices['NDVI']=np.clip(np.nan_to_num(ndvi, nan=0.0, posinf=clip_val, neginf=-clip_val), -clip_val, clip_val)

        denominator_ndwi=green+nir; ndwi=np.full_like(denominator_ndwi, 0.0, dtype=np.float32)
        valid_mask_ndwi=np.abs(denominator_ndwi)>epsilon; ndwi[valid_mask_ndwi]=(green[valid_mask_ndwi]-nir[valid_mask_ndwi])/denominator_ndwi[valid_mask_ndwi]
        indices['NDWI']=np.clip(np.nan_to_num(ndwi, nan=0.0, posinf=clip_val, neginf=-clip_val), -clip_val, clip_val)

        denominator_ndbi=swir1+nir; ndbi=np.full_like(denominator_ndbi, 0.0, dtype=np.float32)
        valid_mask_ndbi=np.abs(denominator_ndbi)>epsilon; ndbi[valid_mask_ndbi]=(swir1[valid_mask_ndbi]-nir[valid_mask_ndbi])/denominator_ndbi[valid_mask_ndbi]
        indices['NDBI']=np.clip(np.nan_to_num(ndbi, nan=0.0, posinf=clip_val, neginf=-clip_val), -clip_val, clip_val)

        denominator_ndre1=nir+re1; ndre1=np.full_like(denominator_ndre1, 0.0, dtype=np.float32)
        valid_mask_ndre1=np.abs(denominator_ndre1)>epsilon; ndre1[valid_mask_ndre1]=(nir[valid_mask_ndre1]-re1[valid_mask_ndre1])/denominator_ndre1[valid_mask_ndre1]
        indices['NDRE1']=np.clip(np.nan_to_num(ndre1, nan=0.0, posinf=clip_val, neginf=-clip_val), -clip_val, clip_val)

    except Exception as e: print(f"Error calculating indices: {e}"); return {}
    return indices

# --- Normalization ---
def normalize_16ch_per_image(image_np_16ch: np.ndarray) -> Optional[np.ndarray]:
    if image_np_16ch is None or image_np_16ch.ndim != 3 or image_np_16ch.shape[0] != 16: return None
    try:
        mean=np.nanmean(image_np_16ch, axis=(1, 2), keepdims=True); std=np.nanstd(image_np_16ch, axis=(1, 2), keepdims=True)
        mean=np.nan_to_num(mean, nan=0.0); std=np.nan_to_num(std, nan=1.0); std[std < 1e-7]=1.0 # Avoid division by zero
        normalized_image=(image_np_16ch - mean) / std
        if np.isnan(normalized_image).any() or np.isinf(normalized_image).any(): normalized_image = np.nan_to_num(normalized_image, nan=0.0, posinf=0.0, neginf=0.0)
        return normalized_image.astype(np.float32)
    except Exception as e: print(f"Error during 16ch normalization: {e}"); return None

# --- Dataset Class ---
class Sentinel2Dataset(Dataset):
    def __init__(self, paths_labels: List[Tuple[str, int]], transform: Optional[Callable] = None, output_channels: int = 16):
        self.paths_labels = [(p, l) for p, l in paths_labels if os.path.exists(p)]
        if len(self.paths_labels) != len(paths_labels): print(f"Warning: Filtered {len(paths_labels) - len(self.paths_labels)} non-existent paths.")
        self.transform = transform
        self.output_channels = output_channels # Should be 16 (12 bands + 4 indices)
        print(f"Initialized Sentinel2Dataset with {len(self.paths_labels)} samples. Output channels: {self.output_channels}")

    def __len__(self): return len(self.paths_labels)

    def __getitem__(self, idx):
        image_path, label = self.paths_labels[idx]
        try:
            image_np_12 = load_sentinel2_image(image_path);
            if image_np_12 is None: return None # Skip if loading failed

            indices_dict = calculate_indices_from_raw(image_np_12);
            if not indices_dict: return None # Skip if index calculation failed

            # Ensure indices are correctly shaped and ordered
            indices_list=[indices_dict['NDVI'], indices_dict['NDWI'], indices_dict['NDBI'], indices_dict['NDRE1']]
            h,w=image_np_12.shape[1], image_np_12.shape[2]; indices_arrays=[idx_arr[np.newaxis, :h, :w] for idx_arr in indices_list]

            try: image_np_16 = np.concatenate([image_np_12] + indices_arrays, axis=0)
            except ValueError as e: print(f"Concat Error {image_path}: {e}, Bands shape: {image_np_12.shape}, Indices shapes: {[arr.shape for arr in indices_arrays]}"); return None

            if image_np_16.shape[0] != self.output_channels: return None # Should be 16 channels

            image_np_norm_16 = normalize_16ch_per_image(image_np_16);
            if image_np_norm_16 is None: return None # Skip if normalization failed

            image_tensor = torch.from_numpy(image_np_norm_16).float() # Ensure float
            if self.transform: image_tensor = self.transform(image_tensor)

            # Final check on tensor shape
            if image_tensor.shape[0]!=self.output_channels or image_tensor.ndim!=3: return None

            label_tensor = torch.tensor(label, dtype=torch.long)
            return image_tensor, label_tensor, image_path # Return path for debugging
        except Exception as e: print(f"Error processing image {image_path}: {e}"); traceback.print_exc(); return None


# --- Data Transforms ---
IMG_SIZE = config["data"]["image_size"]
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.3),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0, inplace=False),
])
val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
])

# --- Create Datasets ---
print("Scanning training directory and creating dataset splits...")
full_dataset_paths_labels = []
class_to_idx_map = {}
class_names = [] # Will store sorted class names
idx_counter = 0
train_root_dir = Path(config["data"]["train_dir"])
if not train_root_dir.exists(): raise FileNotFoundError(f"Training directory not found: {train_root_dir}")

# Sort class folders to ensure consistent class_to_idx mapping
for class_folder in sorted(train_root_dir.iterdir()): # Ensure sorted order
    if class_folder.is_dir() and not class_folder.name.startswith('.'): # Skip hidden files/folders
        class_name = class_folder.name
        if class_name not in class_to_idx_map:
            class_to_idx_map[class_name] = idx_counter
            class_names.append(class_name) # Add to sorted list
            idx_counter += 1
        class_idx = class_to_idx_map[class_name]
        for filepath in class_folder.glob('*.tif'): # Assuming .tif files
            full_dataset_paths_labels.append((str(filepath), class_idx))

num_classes = len(class_names)
if num_classes == 0: raise FileNotFoundError(f"No valid class folders found in {train_root_dir}")
config["model"]["num_classes"] = num_classes
config["model"]["class_names"] = class_names # Store sorted class names in config
print(f"Found {len(full_dataset_paths_labels)} training paths in {num_classes} classes: {class_names}")
print(f"Class mapping: {class_to_idx_map}")


try:
    train_info, val_info = train_test_split(
        full_dataset_paths_labels,
        train_size=float(config["data"]["train_ratio"]), # Cast train_ratio
        random_state=int(config["train"]["seed"]),       # Cast seed
        stratify=[label for _, label in full_dataset_paths_labels] # Stratify by labels
    )
except ValueError as e: # Fallback if stratification fails (e.g., too few samples per class)
    print(f"Warning: Stratified split failed ({e}). Falling back to non-stratified split.")
    train_info, val_info = train_test_split(
        full_dataset_paths_labels,
        train_size=float(config["data"]["train_ratio"]), # Cast train_ratio
        random_state=int(config["train"]["seed"])        # Cast seed
    )


print("Creating Sentinel2Dataset instances...")
train_dataset = Sentinel2Dataset(
    train_info,
    transform=train_transforms,
    output_channels=int(config["model"]["input_channels"]) # Cast channels
)
val_tif_dataset = Sentinel2Dataset( # Renamed to avoid confusion if other val sets exist
    val_info,
    transform=val_transforms,
    output_channels=int(config["model"]["input_channels"]) # Cast channels
)

# --- Create DataLoaders ---
def collate_fn(batch): # Custom collate to handle None from dataset
    batch = [item for item in batch if item is not None] # Filter out None items
    if not batch: return None, None, None # If batch is empty after filtering
    try:
        images = torch.stack([item[0] for item in batch]); labels = torch.stack([item[1] for item in batch]); paths = [item[2] for item in batch]
        return images, labels, paths
    except Exception as e: print(f"Error in collate_fn: {e}. Skipping batch."); return None, None, None


persistent_workers = int(config["data"]["num_workers"]) > 0 # Cast num_workers
train_loader = DataLoader(
    train_dataset,
    batch_size=int(config["data"]["batch_size"]), # Cast batch_size
    shuffle=True,
    num_workers=int(config["data"]["num_workers"]), # Cast num_workers
    pin_memory=True, # If using CUDA
    collate_fn=collate_fn, # Use custom collate
    drop_last=True, # Important for consistent batch sizes, esp. with SWA BN update
    persistent_workers=persistent_workers
)
val_loader = DataLoader(
    val_tif_dataset,
    batch_size=int(config["data"]["batch_size"])*2, # Cast batch_size, often larger for val
    shuffle=False,
    num_workers=int(config["data"]["num_workers"]), # Cast num_workers
    pin_memory=True,
    collate_fn=collate_fn, # Use custom collate
    persistent_workers=persistent_workers
)
print("Train and Validation DataLoaders created.")


# --- Model Definition (Handles Pretrained and Scratch based on config) ---
def adapt_resnet_for_multichannel(model_name="resnet50", pretrained=False, num_classes=10, input_channels=16):
    """Loads ResNet architecture, optionally loads pretrained weights, adapts layers."""
    print(f"Loading {model_name} architecture...")
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    if pretrained: print("Attempting to load PRETRAINED weights (ImageNet)...")
    else: print("Using RANDOM weights (training from scratch)...")

    model = models.get_model(model_name, weights=weights) # Use new API

    # Adapt the first convolutional layer
    original_conv1 = model.conv1
    new_conv1 = nn.Conv2d(input_channels, original_conv1.out_channels,
                          kernel_size=original_conv1.kernel_size,
                          stride=original_conv1.stride,
                          padding=original_conv1.padding,
                          bias=(original_conv1.bias is not None))

    if pretrained:
        print(f"Adapting pretrained weights for conv1 from 3 to {input_channels} channels...")
        original_weights = original_conv1.weight.data # (out_channels, 3, K, K)
        # Average weights across the 3 input channels
        avg_weights = torch.mean(original_weights, dim=1, keepdim=True) # (out_channels, 1, K, K)
        # Repeat these averaged weights for the new number of input channels
        repeated_weights = avg_weights.repeat(1, input_channels, 1, 1) # (out_channels, input_channels, K, K)
        new_conv1.weight.data = repeated_weights
        if new_conv1.bias is not None: new_conv1.bias.data = original_conv1.bias.data # Copy bias if exists
        print("Pretrained conv1 weights adapted.")
    else: # Initialize weights for training from scratch
        print("Initializing conv1 weights randomly...")
        nn.init.kaiming_normal_(new_conv1.weight, mode='fan_out', nonlinearity='relu')
        if new_conv1.bias is not None: nn.init.zeros_(new_conv1.bias)
        print("Random conv1 weights initialized.")

    model.conv1 = new_conv1 # Replace the first conv layer

    # Adapt the final fully connected layer
    print("Replacing and initializing final fc layer...")
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    nn.init.xavier_uniform_(model.fc.weight); # Initialize new fc layer
    if model.fc.bias is not None: nn.init.zeros_(model.fc.bias)
    print(f"Replaced model.fc layer for {num_classes} classes.")
    return model

# --- MixUp and CutMix Implementations (Unchanged) ---
def mixup_data(x, y, alpha=1.0, device='cuda'):
    if alpha > 0: lam = np.random.beta(alpha, alpha)
    else: lam = 1
    batch_size = x.size()[0]; index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]; y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def rand_bbox(size, lam): # size: (B, C, H, W)
    W = size[2]; H = size[3]; cut_rat = np.sqrt(1. - lam); cut_w = int(W * cut_rat); cut_h = int(H * cut_rat)
    # Uniformly sample top-left corner
    cx = np.random.randint(W); cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W); bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W); bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def cutmix_data(x, y, alpha=1.0, device='cuda'):
    if alpha > 0: lam = np.random.beta(alpha, alpha) # lambda for bbox size
    else: lam = 1 # No cutmix
    batch_size = x.size()[0]; index = torch.randperm(batch_size).to(device)
    y_a, y_b = y, y[index]; bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2] # Apply cutmix
    # Adjust lambda to reflect the actual mixed area
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, y_a, y_b, lam


# --- Learning Rate Warm-up Helper ---
def get_lr(optimizer):
    if optimizer is None: return 0.0
    for param_group in optimizer.param_groups: return param_group['lr']

def adjust_lr_with_warmup(optimizer, initial_lr, epoch, step, steps_per_epoch, warmup_epochs, config):
    # epoch is 0-indexed for calculation here
    total_warmup_steps = warmup_epochs * steps_per_epoch
    current_step = epoch * steps_per_epoch + step # current step within warmup phase

    if current_step < total_warmup_steps:
        # Linear warm-up from a very small LR to initial_lr
        start_lr = float(config['train'].get('warmup_start_lr', 1e-7)) # Configurable start LR for warmup
        lr = start_lr + (initial_lr - start_lr) * (current_step + 1) / total_warmup_steps
        for param_group in optimizer.param_groups: param_group['lr'] = lr
        return lr # Return the adjusted LR
    else:
        # Warm-up finished, LR is controlled by main scheduler or is fixed at initial_lr
        return None # Indicate no adjustment made by this function

# --- Training/Validation Epoch Helper (with casting fix) ---
def run_epoch(model, loader, criterion, optimizer, scaler, scheduler, device, is_training,
              epoch_num, total_epochs, config, steps_per_epoch, is_cosine_scheduler=False):
    # --- Explicit Type Casting for Robustness ---
    initial_lr = float(config['train']['lr'])
    warmup_epochs = int(config['train']['warmup_epochs'])
    gradient_accumulation_steps = int(config['train']['gradient_accumulation_steps'])
    # Handle optional gradient_clip_norm correctly
    gradient_clip_norm_config = config['train'].get("gradient_clip_norm")
    gradient_clip_norm = float(gradient_clip_norm_config) if gradient_clip_norm_config is not None else None

    amp_enabled = scaler is not None and bool(config['amp_enabled']) and (device.type == 'cuda') # Cast amp_enabled, ensure CUDA for AMP
    use_mixup = bool(config['train']['use_mixup']) and is_training
    use_cutmix = bool(config['train']['use_cutmix']) and is_training
    mixup_alpha = float(config['train']['mixup_alpha'])
    cutmix_alpha = float(config['train']['cutmix_alpha'])
    mixup_prob = float(config['train']['mixup_prob']) # Probability of applying either MixUp or CutMix
    label_smoothing = float(config['train']['label_smoothing'])
    # --- End Type Casting ---

    if is_training: model.train(); print(f'---> Training Epoch {epoch_num}/{total_epochs} | Target LR: {initial_lr:.4e}')
    else: model.eval(); print(f'---> Validation Epoch {epoch_num}/{total_epochs}')

    running_loss = 0.0; correct_predictions = 0; total_samples = 0; start_time = time.time()
    # Base criterion for MixUp/CutMix if label smoothing is used
    base_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing if is_training else 0.0, reduction='mean')


    progress_bar = tqdm(loader, desc=f"Epoch {epoch_num} {'Train' if is_training else 'Val'}", leave=False)

    # Ensure optimizer is zero_grad at the start of an epoch if training
    if is_training and optimizer:
        optimizer.zero_grad(set_to_none=True)


    with torch.set_grad_enabled(is_training): # Context manager for gradients
        for batch_idx, batch_data in enumerate(progress_bar):
            if batch_data is None or batch_data[0] is None: continue # Skip bad batches from collate_fn
            inputs, targets, _ = batch_data # paths are also returned by collate_fn
            inputs, targets_orig = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

            current_lr = None # For display
            # Warm-up LR adjustment (epoch_num is 1-indexed, so use epoch_num-1 for 0-indexed calculation)
            # Pass integer warmup_epochs here
            if is_training and warmup_epochs > 0 and (epoch_num-1) < warmup_epochs : # epoch_num is 1-based
                # Pass 0-indexed epoch (epoch_num-1) to adjust_lr_with_warmup
                current_lr = adjust_lr_with_warmup(optimizer, initial_lr, epoch_num-1, batch_idx, steps_per_epoch, warmup_epochs, config)

            # MixUp/CutMix Augmentation
            apply_mixup_cutmix = (use_mixup or use_cutmix) and random.random() < mixup_prob
            if apply_mixup_cutmix:
                use_this_mixup = use_mixup # Default to mixup if only one is enabled
                if use_mixup and use_cutmix: use_this_mixup = (random.random() < 0.5) # 50/50 if both enabled

                if use_this_mixup: inputs, targets_a, targets_b, lam = mixup_data(inputs, targets_orig, mixup_alpha, device)
                else: inputs, targets_a, targets_b, lam = cutmix_data(inputs, targets_orig, cutmix_alpha, device)
            else: # No MixUp/CutMix for this batch
                apply_mixup_cutmix = False; targets_a, targets_b, lam = None, None, None


            # AMP autocast for forward pass
            # Use integer gradient_accumulation_steps
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                outputs = model(inputs)
                if apply_mixup_cutmix: loss = mixup_criterion(base_criterion, outputs, targets_a, targets_b, lam)
                else: loss = criterion(outputs, targets_orig) # Use main criterion (can also have LS)

                if is_training and gradient_accumulation_steps > 1: loss = loss / gradient_accumulation_steps


            if is_training:
                if amp_enabled: scaler.scale(loss).backward() # AMP backward
                else: loss.backward() # Standard backward

                # Use integer gradient_accumulation_steps
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    if amp_enabled: scaler.unscale_(optimizer); # Unscale before clipping

                    # Use float gradient_clip_norm if not None
                    if gradient_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)

                    if amp_enabled: scaler.step(optimizer); scaler.update() # AMP optimizer step
                    else: optimizer.step() # Standard optimizer step

                    optimizer.zero_grad(set_to_none=True) # Zero gradients for next accumulation cycle

                    # Scheduler step (per batch/step for CosineAnnealingLR after warmup)
                    # Use integer warmup_epochs
                    is_after_warmup = (warmup_epochs == 0) or ((epoch_num-1) >= warmup_epochs)
                    if is_after_warmup and is_cosine_scheduler and scheduler is not None:
                        # Cosine scheduler steps per effective optimizer step
                        scheduler.step()


            # Loss and Accuracy Calculation
            if torch.isnan(loss) or torch.isinf(loss): print(f"\nWARNING: NaN/Inf loss {loss.item()} E{epoch_num} B{batch_idx+1}. Skipping."); optimizer.zero_grad(set_to_none=True); continue

            # Use integer gradient_accumulation_steps
            batch_loss = loss.item() * gradient_accumulation_steps if is_training and gradient_accumulation_steps > 1 else loss.item()
            running_loss += batch_loss * inputs.size(0) # Accumulate loss scaled by batch size
            _, predicted = torch.max(outputs.data, 1); total_samples += targets_orig.size(0)
            correct_predictions += (predicted == targets_orig).sum().item()

            current_acc = correct_predictions / total_samples if total_samples > 0 else 0
            if is_training: display_lr = get_lr(optimizer); progress_bar.set_postfix(loss=f"{batch_loss:.4f}", acc=f"{current_acc:.3f}", lr=f"{display_lr:.2e}")
            else: progress_bar.set_postfix(loss=f"{batch_loss:.4f}", acc=f"{current_acc:.3f}")


    epoch_duration = time.time() - start_time
    if total_samples == 0: print(f"Warning: No valid samples processed in epoch {epoch_num}."); return 0.0, 0.0 # Handle empty epoch

    epoch_loss = running_loss / total_samples; epoch_acc = correct_predictions / total_samples
    mode_str = "Training" if is_training else "Validation"
    final_lr = None # Only relevant for training
    if is_training: final_lr = get_lr(optimizer); print(f'\n    Epoch {epoch_num} {mode_str} Summary: Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f} | Final LR: {final_lr:.4e} | Duration: {epoch_duration:.2f}s')
    else: print(f'\n    Epoch {epoch_num} {mode_str} Summary: Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f} | Duration: {epoch_duration:.2f}s')

    return epoch_loss, epoch_acc

# --- Plotting Functions (NEW) ---
def plot_training_curves(history: Dict[str, List[float]], save_dir: Path, model_name: str):
    """Plots and saves training and validation loss and accuracy curves."""
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(15, 6))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'bo-', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss')
    plt.title(f'{model_name} - Loss over Epochs', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'bo-', label='Train Accuracy')
    plt.plot(epochs, history['val_acc'], 'ro-', label='Validation Accuracy')
    plt.title(f'{model_name} - Accuracy over Epochs', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.tight_layout(pad=3.0)
    plot_path = save_dir / f"{model_name}_training_curves.png"
    try:
        plt.savefig(plot_path, dpi=300)
        print(f"Training curves saved to {plot_path}")
    except Exception as e:
        print(f"Error saving training curves plot: {e}")
    plt.close()


def plot_confusion_matrix_custom(y_true: List[int], y_pred: List[int], class_names: List[str],
                                 save_dir: Path, model_name: str, title_suffix: str = ""):
    """Computes, plots, and saves a confusion matrix and prints classification report."""
    if not y_true or not y_pred:
        print(f"Cannot generate confusion matrix for {model_name}{title_suffix}: No data provided.")
        return

    cm = sk_confusion_matrix(y_true, y_pred, labels=list(range(len(class_names)))) # Ensure all classes are considered

    plt.figure(figsize=(max(8, len(class_names)//1.5), max(6, len(class_names)//2))) # Adjust size based on num_classes
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 10}, cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - {model_name}{title_suffix}', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)

    plot_filename = f"{model_name}_confusion_matrix{title_suffix.replace(' ', '_').replace('(', '').replace(')', '')}.png"
    plot_path = save_dir / plot_filename
    try:
        plt.tight_layout(pad=1.0)
        plt.savefig(plot_path, dpi=300)
        print(f"Confusion matrix saved to {plot_path}")
    except Exception as e:
        print(f"Error saving confusion matrix plot: {e}")
    plt.close()

    # Print Classification Report
    print(f"\nClassification Report - {model_name}{title_suffix}:")
    try:
        report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0, labels=list(range(len(class_names))))
        print(report)
    except Exception as e:
        print(f"Error generating classification report: {e}")


# --- Evaluation function for collecting predictions (NEW) ---
def evaluate_model_for_plots(model: nn.Module, loader: DataLoader, device: torch.device,
                             num_classes: int, config_amp_enabled: bool):
    """Evaluates the model on the given loader and returns all targets, predictions, loss, and accuracy."""
    model.eval() # Set model to evaluation mode
    all_preds_list = []
    all_targets_list = []
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    # Use a simple criterion for evaluation, label smoothing is not typically used here.
    criterion = nn.CrossEntropyLoss()

    # Determine if AMP should be enabled for evaluation
    amp_eval_enabled = bool(config_amp_enabled) and (device.type == 'cuda')

    progress_bar = tqdm(loader, desc="Evaluating for Plots", leave=False, unit="batch")
    with torch.no_grad(): # Disable gradient calculations
        for batch_data in progress_bar:
            if batch_data is None or batch_data[0] is None:
                print("Skipping a None batch in evaluate_model_for_plots.")
                continue # Skip if collate_fn returned None

            inputs, targets, _ = batch_data # We don't need paths here
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

            # AMP autocast for the forward pass if enabled
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_eval_enabled):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted_classes = torch.max(outputs.data, 1) # Get the predicted class indices
            total_samples += targets.size(0)
            correct_predictions += (predicted_classes == targets).sum().item()

            all_preds_list.extend(predicted_classes.cpu().numpy())
            all_targets_list.extend(targets.cpu().numpy())

            current_acc_eval = correct_predictions / total_samples if total_samples > 0 else 0
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{current_acc_eval:.3f}")

    if total_samples == 0:
        print("Warning: No samples were processed during evaluation for plots. Returning empty lists.")
        return [], [], 0.0, 0.0

    avg_loss = running_loss / total_samples
    avg_acc = correct_predictions / total_samples
    print(f"\nEvaluation for Plots Summary: Average Loss: {avg_loss:.4f}, Average Accuracy: {avg_acc:.4f}")
    return all_targets_list, all_preds_list, avg_loss, avg_acc


# --- Main Execution Logic ---
if __name__ == '__main__':
    # Paths from config
    base_save_path = Path(config["model"]["base_save_path"])
    base_save_path.mkdir(parents=True, exist_ok=True) # Ensure save directory exists
    model_base_name = config['model']['name']
    best_model_path = base_save_path / f"{model_base_name}_best_non_swa.pth"
    swa_model_path = base_save_path / f"{model_base_name}_swa_final.pth"


    if config["model"]["num_classes"] is None: raise ValueError("Num classes not determined prior to model instantiation.")
    num_classes = int(config["model"]["num_classes"]) # Cast num_classes
    # Ensure class_names are loaded if needed for plots later
    class_names_from_config = config["model"].get("class_names")
    if not class_names_from_config or len(class_names_from_config) != num_classes:
        print(f"Warning: class_names in config mismatch num_classes. Using generic names for plots if needed.")
        class_names_for_plots = [f"Class {i}" for i in range(num_classes)]
    else:
        class_names_for_plots = class_names_from_config


    # Instantiate model using config
    model = adapt_resnet_for_multichannel(
        model_name=config["model"].get("architecture", "resnet50"), # Default to resnet50 if not specified
        pretrained=bool(config["model"]["pretrained"]), # Cast pretrained
        num_classes=num_classes,
        input_channels=int(config["model"]["input_channels"]) # Cast channels
    )
    model.to(DEVICE)

    # SWA Model Initialization
    # The SWA model will average the weights of the 'model'
    swa_model = AveragedModel(model, device=DEVICE) # device can be specified here
    print("SWA Model wrapper created.")

    # Optimizer and Loss from config
    criterion = nn.CrossEntropyLoss(label_smoothing=float(config["train"]["label_smoothing"])) # Cast smoothing
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["lr"]),           # Cast lr
        weight_decay=float(config["train"]["weight_decay"]) # Cast wd
    )
    print(f"Optimizer: AdamW (Target Initial LR={config['train']['lr']}, WD={config['train']['weight_decay']})")

    # Cast warmup_epochs
    warmup_epochs = int(config["train"]["warmup_epochs"])
    if warmup_epochs > 0:
        print(f"Using {warmup_epochs} warm-up epochs.")
        # Set initial LR very low for warmup phase
        warmup_start_lr = float(config['train'].get('warmup_start_lr', 1e-7))
        for param_group in optimizer.param_groups: param_group['lr'] = warmup_start_lr


    # Scheduler Setup from config
    num_epochs = int(config["train"]["epochs"]) # Cast epochs
    gradient_accumulation_steps = int(config["train"]["gradient_accumulation_steps"]) # Cast steps
    if gradient_accumulation_steps <= 0: gradient_accumulation_steps = 1 # Ensure positive
    gradient_clip_norm_config = config["train"].get("gradient_clip_norm") # Handle None case
    gradient_clip_norm = float(gradient_clip_norm_config) if gradient_clip_norm_config is not None else None
    if gradient_clip_norm: print(f"Using gradient clipping with max_norm={gradient_clip_norm}")


    # Calculate steps_per_epoch for schedulers (considering accumulation)
    # This is the number of optimizer steps per epoch
    if len(train_loader) > 0 :
        steps_per_epoch = len(train_loader) // gradient_accumulation_steps
        if steps_per_epoch == 0: steps_per_epoch = 1 # Handle small datasets
    else:
        print("Warning: train_loader is empty. Setting steps_per_epoch to 1.")
        steps_per_epoch = 1


    # Main LR Scheduler (Non-SWA phase)
    swa_start_epoch = int(config["train"]["swa_start_epoch"]) # Cast swa start epoch
    # Number of epochs before SWA starts (1-indexed)
    epochs_before_swa = swa_start_epoch - 1
    # Total optimizer steps available for the main scheduler after warmup
    # Ensure this is positive. If swa_start_epoch <= warmup_epochs, T_max might be non-positive.
    steps_before_swa = max(1, steps_per_epoch * (epochs_before_swa - warmup_epochs))


    scheduler = None
    is_cosine_scheduler = False # Flag to indicate if CosineAnnealingLR is used (steps per batch)
    if config["train"]["scheduler"] == "CosineAnnealingLR":
        eta_min_config = config["train"]["eta_min"]
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=steps_before_swa, # Total steps for the cosine cycle (after warmup)
            eta_min=float(eta_min_config) # Cast eta_min
        )
        is_cosine_scheduler = True
        print(f"Using CosineAnnealingLR scheduler until SWA starts (T_max_steps={steps_before_swa} after warm-up, eta_min={float(eta_min_config):.2e})")
    elif config["train"]["scheduler"] == "ReduceLROnPlateau":
        # Add casting for ReduceLROnPlateau patience if needed, assume it's int for now
        patience_plateau = int(config["train"].get("scheduler_patience", 7)) # Example optional key
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience_plateau, verbose=True)
        is_cosine_scheduler = False; print("Using ReduceLROnPlateau scheduler.")
    else:
        print(f"Warning: Unknown scheduler type '{config['train']['scheduler']}'. No main scheduler used before SWA.")


    # SWA Scheduler
    swa_lr_config = config["train"]["swa_lr"]
    anneal_epochs_config = config["train"]["swa_anneal_epochs"]
    # SWA scheduler steps per epoch, not per batch
    swa_scheduler = SWALR(
        optimizer,
        swa_lr=float(swa_lr_config),           # Cast swa_lr
        anneal_epochs=int(anneal_epochs_config), # Cast anneal_epochs
        anneal_strategy='cos' # Can be 'linear' or 'cos'
    )
    print(f"SWA will start at epoch {swa_start_epoch} with SWA LR {float(swa_lr_config):.2e}, anneal over {anneal_epochs_config} epochs.")


    # Training Loop Setup
    patience = int(config["train"]["patience"]) # Cast patience
    checkpoint_interval = int(config["train"]["checkpoint_interval"]) # Cast interval
    overall_best_val_loss = float('inf'); total_epochs_run = 0
    # Scaler for AMP, enabled based on config and device
    amp_enabled_main = bool(config['amp_enabled']) and (DEVICE.type == 'cuda')
    scaler = GradScaler(enabled=amp_enabled_main)
    if scaler.is_enabled(): print("Using Automatic Mixed Precision (AMP) for training.")
    else: print("AMP is disabled for training.")


    print(f"\n--- Starting Training for {num_epochs} epochs ---")
    epochs_without_improvement = 0

    # History tracking for plots (NEW)
    training_history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }


    # --- Training Loop ---
    for epoch in range(num_epochs): # epoch is 0-indexed here
        epoch_num = epoch + 1 # 1-indexed for display and SWA logic
        total_epochs_run = epoch_num # Track total epochs completed

        # Determine which scheduler to use for this epoch
        use_swa_scheduler_for_lr_setting = epoch_num >= swa_start_epoch # SWA LR setting active
        active_scheduler_for_step = None # Scheduler that needs .step()
        is_active_cosine = False

        if not use_swa_scheduler_for_lr_setting: # Before SWA phase
            # LR is either from warmup or main scheduler (Cosine or Plateau)
            active_scheduler_for_step = scheduler # This will be Cosine or Plateau
            is_active_cosine = is_cosine_scheduler
        # If use_swa_scheduler_for_lr_setting is true, SWA scheduler will be stepped at end of epoch.
        # LR during SWA phase is managed by SWALR.

        if use_swa_scheduler_for_lr_setting and epoch_num == swa_start_epoch:
            print(f"--- Epoch {epoch_num}: SWA phase begins. SWA LR schedule active. Averaging model weights starts now. ---")
            # Optimizer's LR will now be controlled by swa_scheduler.step() at end of epoch.

        # Run training epoch
        # Pass the full config dict
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, optimizer, scaler,
            active_scheduler_for_step if not use_swa_scheduler_for_lr_setting else None, # Pass main scheduler only if not in SWA phase for LR setting
            DEVICE, True, epoch_num, num_epochs, config, steps_per_epoch,
            is_active_cosine # Pass flag for Cosine scheduler's per-step behavior
        )

        # Run validation epoch
        avg_val_loss, val_accuracy = run_epoch(
            model, val_loader, criterion, None, scaler, None, # No optimizer/scheduler needed for val
            DEVICE, False, epoch_num, num_epochs, config, steps_per_epoch, False
        )

        # Store metrics for plotting (NEW)
        training_history['train_loss'].append(train_loss)
        training_history['train_acc'].append(train_accuracy)
        training_history['val_loss'].append(avg_val_loss)
        training_history['val_acc'].append(val_accuracy)


        # Step epoch-based schedulers (ReduceLROnPlateau, SWALR)
        # CosineAnnealingLR is stepped per optimizer step inside run_epoch (if active)
        is_after_warmup = (warmup_epochs == 0) or (epoch_num > warmup_epochs)

        if is_after_warmup:
            if use_swa_scheduler_for_lr_setting:
                swa_scheduler.step() # SWA scheduler steps per epoch
            elif not is_cosine_scheduler and scheduler is not None: # e.g., ReduceLROnPlateau
                scheduler.step(avg_val_loss) # Plateau scheduler steps with validation loss


        # SWA Model Weight Averaging
        if epoch_num >= swa_start_epoch:
            swa_model.update_parameters(model) # Average weights into swa_model
            if epoch_num == swa_start_epoch:
                 print(f"--- Epoch {epoch_num}: SWA model updated with current model weights for the first time. ---")


        # Checkpointing based on non-SWA model validation loss
        is_best = avg_val_loss < overall_best_val_loss
        if is_best:
            overall_best_val_loss = avg_val_loss; epochs_without_improvement = 0
            try:
                torch.save(model.state_dict(), best_model_path)
                print(f'---> Best non-SWA model state saved to {best_model_path} (Val Loss: {avg_val_loss:.4f})')
            except Exception as e: print(f"Error saving best non-SWA model state_dict: {e}")
        else:
            epochs_without_improvement += 1
            print(f'---> Val loss did not improve for {epochs_without_improvement} epochs from best: {overall_best_val_loss:.4f}. Current: {avg_val_loss:.4f}')


        # Periodic checkpoint saving (non-SWA model)
        if checkpoint_interval > 0 and epoch_num % checkpoint_interval == 0:
            periodic_path = base_save_path / f"{model_base_name}_epoch_{epoch_num}.pth"
            try:
                torch.save(model.state_dict(), periodic_path)
                print(f'---> Periodic checkpoint saved to {periodic_path}')
            except Exception as e: print(f"Error saving periodic checkpoint: {e}")


        # Early stopping check
        if epochs_without_improvement >= patience:
            print(f'Early stopping triggered at epoch {epoch_num} due to no improvement for {patience} epochs.')
            break
        # total_epochs_run is already updated at the start of the loop

    # --- After Training Loop ---
    print("\n--- Training Complete ---")
    print(f"Total epochs run: {total_epochs_run}")

    # Plot training curves (NEW)
    if training_history['train_loss']: # Check if any history was recorded
        plot_training_curves(training_history, base_save_path, model_base_name)
    else:
        print("No training history recorded, skipping curve plots.")


    # Evaluate best non-SWA model for confusion matrix (NEW)
    if best_model_path.exists():
        print(f"\n--- Evaluating best non-SWA model ({best_model_path.name}) for plots ---")
        # Re-instantiate model structure for loading state_dict
        # Set pretrained=False as we are loading our own weights, not ImageNet's again
        best_model_to_eval = adapt_resnet_for_multichannel(
            model_name=config["model"].get("architecture", "resnet50"),
            pretrained=False, # IMPORTANT: Set to False when loading custom weights
            num_classes=num_classes,
            input_channels=int(config["model"]["input_channels"])
        )
        try:
            best_model_to_eval.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
            best_model_to_eval.to(DEVICE) # Move to device

            val_targets_non_swa, val_preds_non_swa, _, _ = evaluate_model_for_plots(
                best_model_to_eval, val_loader, DEVICE, num_classes, config['amp_enabled'] # Pass amp_enabled from config
            )
            if val_targets_non_swa and val_preds_non_swa: # Check if lists are not empty
                plot_confusion_matrix_custom(
                    val_targets_non_swa, val_preds_non_swa,
                    class_names_for_plots, base_save_path,
                    model_base_name, title_suffix=" (Best Non-SWA)"
                )
        except Exception as e:
            print(f"Error evaluating best non-SWA model or plotting its confusion matrix: {e}")
            traceback.print_exc()
    else:
        print(f"Best non-SWA model path not found: {best_model_path}. Skipping its evaluation for plots.")


    # SWA model processing and evaluation for confusion matrix (NEW)
    if total_epochs_run >= swa_start_epoch: # Check if SWA phase was active
        print("\n--- Processing and Evaluating SWA model ---")
        # Update SWA Batch Norm statistics using the training data loader
        # This should be done ONCE after SWA updates and before final SWA model saving/evaluation
        print("Updating SWA Batch Norm statistics using train_loader...")
        bn_update_device = DEVICE # Use the main device for BN update
        try:
            # Ensure train_loader is not empty for BN update
            if len(train_loader) > 0:
                 # It's crucial that swa_model is on the correct device before update_bn
                swa_model.to(bn_update_device) # Ensure swa_model itself is on the device
                torch.optim.swa_utils.update_bn(train_loader, swa_model, device=bn_update_device)
                print("SWA Batch Norm update complete.")
            else:
                print("Warning: train_loader is empty. Skipping SWA Batch Norm update.")


            # Evaluate the SWA model (which now has updated BN stats)
            print("Evaluating SWA model for plots...")
            # swa_model is an AveragedModel, it will call its 'module' (the actual model) internally.
            val_targets_swa, val_preds_swa, swa_val_loss, swa_val_acc = evaluate_model_for_plots(
                swa_model, val_loader, DEVICE, num_classes, config['amp_enabled'] # Pass amp_enabled from config
            )
            print(f"SWA Model Final Validation: Loss: {swa_val_loss:.4f}, Accuracy: {swa_val_acc:.4f}")

            if val_targets_swa and val_preds_swa: # Check if lists are not empty
                plot_confusion_matrix_custom(
                    val_targets_swa, val_preds_swa,
                    class_names_for_plots, base_save_path,
                    model_base_name, title_suffix=" (SWA Final)"
                )

            # Save the final SWA model (module's state_dict) AFTER BN update
            print(f"Saving final SWA model state (after BN update) to {swa_model_path}")
            torch.save(swa_model.module.state_dict(), swa_model_path) # Save the underlying averaged model's state_dict
            print(f"Final SWA model saved to {swa_model_path}")

        except Exception as e:
            print(f"Error during SWA model processing (BN update, evaluation, saving, or plotting): {e}")
            traceback.print_exc()
    else:
        print(f"SWA phase (start epoch {swa_start_epoch}) was not reached or completed ({total_epochs_run} epochs run). Skipping SWA model finalization and evaluation for plots.")


    if overall_best_val_loss != float('inf'):
        print(f'\nBest non-SWA model was saved to {best_model_path} (Achieved Val Loss: {overall_best_val_loss:.4f})')
    else:
        print("\nNo best non-SWA model was saved (initial validation loss was not improved).")

    print("\n--- Script Finished ---")
    print(f"IMPORTANT: Evaluate the desired checkpoint (e.g., {best_model_path.name} or {swa_model_path.name}) using a separate evaluation script if needed.")
    print(f"Plots saved in: {base_save_path}")

