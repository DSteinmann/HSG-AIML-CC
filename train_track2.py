import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models  # Added for pre-trained models
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

# --- Device Setup ---
USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

# --- Suppress warnings ---
warnings.filterwarnings("ignore")

# --- Configuration ---
CONFIG = {
    "model": {
        "name": "ResNet50_16ch_EuroSAT_v1",  # Updated model name
        "pretrained": True,  # Specify using pretrained weights
        "input_channels": 16,  # 12 bands + 4 indices
        "base_save_path": Path('./outputs/resnet50_16ch_eurosat_v1'),  # Base path for saving checkpoints
        "num_classes": None,  # To be determined from data
        "class_names": None,  # To be determined from data
    },
    "data": {
        "train_dir": Path('./ds/images/remote_sensing/otherDatasets/sentinel_2/tif'),
        "image_size": 224,  # ResNet typically uses 224x224
        "batch_size": 32,  # Adjust based on GPU memory
        "num_workers": 8,
        "train_ratio": 0.9,
    },
    "train": {
        "seed": 1337,
        "epochs": 30,  # Fine-tuning might require fewer epochs initially
        "lr": 1e-4,  # Lower LR for fine-tuning is common
        "optimizer": "AdamW",
        "weight_decay": 1e-4,  # WD might be adjusted for fine-tuning
        # Removed custom head dropout here, will add in model definition if needed
        "scheduler": "CosineAnnealingLR",
        "T_max": 30,  # Match epochs
        "eta_min": 1e-6,
        "patience": 10,  # Adjust patience for fine-tuning
        "label_smoothing": 0.1,
        "gradient_accumulation_steps": 2,
        "gradient_clip_norm": 1.0,
        "use_mixup": True,  # Keep MixUp
        "mixup_alpha": 0.4,
        "use_cutmix": False,
        "cutmix_alpha": 1.0,
        "mixup_prob": 1.0,  # Use MixUp always if enabled
        "checkpoint_interval": 5,
        # Add specific fine-tuning params if needed (e.g., freeze backbone epochs)
        "freeze_backbone_epochs": 0,  # Set > 0 to freeze backbone initially
    },
    "device": DEVICE.type,
    "amp_enabled": USE_CUDA,
}


# --- Seed Setting ---
def set_seed(seed: int):
    random.seed(seed);
    np.random.seed(seed);
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to {seed}")


set_seed(CONFIG["train"]["seed"])

# --- Data Loading and Preprocessing ---
TARGET_BANDS_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12]  # Indices for B1..B9, B11, B12, B8A (omits B10)
# Mapping based on the 12 selected bands (0-based index)
BAND_MAP_12 = {
    "B1_Coastal": 0,
    "B2_Blue": 1,
    "B3_Green": 2,
    "B4_Red": 3,
    "B5_RE1": 4,    # Red Edge 1
    "B6_RE2": 5,    # Red Edge 2
    "B7_RE3": 6,    # Red Edge 3
    "B8_NIR": 7,    # Near Infrared
    "B8A_NIR2": 8,   # Narrow Near Infrared (at index 8)
    "B9_WV": 9,     # Water Vapour (at index 9)
    "B11_SWIR1": 10,  # Short Wave Infrared 1 (at index 10)
    "B12_SWIR2": 11   # Short Wave Infrared 2 (at index 11)
}

def load_sentinel2_image(filepath: str) -> Optional[np.ndarray]:
    """Loads a 13-band TIF file, selects 12 specific bands, returns (12, H, W) NumPy array or None."""
    global TARGET_BANDS_INDICES
    try:
        filepath_lower = filepath.lower()
        if not filepath_lower.endswith(('.tif', '.tiff')): return None
        with rasterio.open(filepath) as src:
            if src.count == 13:
                all_bands = src.read(list(range(1, 14))); image_data_12 = all_bands[TARGET_BANDS_INDICES, :, :]
            else:
                print(f"Error: Expected 13 bands in TIF {filepath}, found {src.count}. Skipping."); return None
        if image_data_12 is None or image_data_12.shape[0] != 12: return None
        # Convert to float32 early on
        return image_data_12.astype(np.float32)
    except Exception as e:
        print(f"Error loading image {filepath}: {e}"); traceback.print_exc(); return None


# --- Index Calculation (operates on UN-NORMALIZED 12 bands) ---
def calculate_indices_from_raw(image_np_12bands: np.ndarray, epsilon=1e-7) -> Dict[str, np.ndarray]:
    """Calculates NDVI, NDWI, NDBI, NDRE1 from an UN-NORMALIZED 12-band NumPy array. Clips output."""
    indices = {};
    clip_val = 1.0  # Standard index range clipping
    global BAND_MAP_12
    try:
        # Get bands from the UN-NORMALIZED 12-band input using the 0-11 mapping
        nir = image_np_12bands[BAND_MAP_12["B8_NIR"], :, :]
        red = image_np_12bands[BAND_MAP_12["B4_Red"], :, :]
        green = image_np_12bands[BAND_MAP_12["B3_Green"], :, :]
        swir1 = image_np_12bands[BAND_MAP_12["B11_SWIR1"], :, :]
        re1 = image_np_12bands[BAND_MAP_12["B5_RE1"], :, :]  # Band B5_RE1

        # NDVI = (NIR - Red) / (NIR + Red)
        denominator_ndvi = nir + red;
        ndvi = np.full_like(denominator_ndvi, 0.0, dtype=np.float32)
        valid_mask_ndvi = np.abs(denominator_ndvi) > epsilon
        ndvi[valid_mask_ndvi] = (nir[valid_mask_ndvi] - red[valid_mask_ndvi]) / denominator_ndvi[valid_mask_ndvi]
        indices['NDVI'] = np.clip(np.nan_to_num(ndvi, nan=0.0, posinf=clip_val, neginf=-clip_val), -clip_val, clip_val)

        # NDWI (Green - NIR) / (Green + NIR) --- McFeeters
        denominator_ndwi = green + nir;
        ndwi = np.full_like(denominator_ndwi, 0.0, dtype=np.float32)
        valid_mask_ndwi = np.abs(denominator_ndwi) > epsilon
        ndwi[valid_mask_ndwi] = (green[valid_mask_ndwi] - nir[valid_mask_ndwi]) / denominator_ndwi[valid_mask_ndwi]
        indices['NDWI'] = np.clip(np.nan_to_num(ndwi, nan=0.0, posinf=clip_val, neginf=-clip_val), -clip_val, clip_val)

        # NDBI (SWIR1 - NIR) / (SWIR1 + NIR)
        denominator_ndbi = swir1 + nir;
        ndbi = np.full_like(denominator_ndbi, 0.0, dtype=np.float32)
        valid_mask_ndbi = np.abs(denominator_ndbi) > epsilon
        ndbi[valid_mask_ndbi] = (swir1[valid_mask_ndbi] - nir[valid_mask_ndbi]) / denominator_ndbi[valid_mask_ndbi]
        indices['NDBI'] = np.clip(np.nan_to_num(ndbi, nan=0.0, posinf=clip_val, neginf=-clip_val), -clip_val, clip_val)

        # NDRE1 (NIR - RE1) / (NIR + RE1) --- Using B5 (RE1)
        denominator_ndre1 = nir + re1;
        ndre1 = np.full_like(denominator_ndre1, 0.0, dtype=np.float32)
        valid_mask_ndre1 = np.abs(denominator_ndre1) > epsilon
        ndre1[valid_mask_ndre1] = (nir[valid_mask_ndre1] - re1[valid_mask_ndre1]) / denominator_ndre1[valid_mask_ndre1]
        indices['NDRE1'] = np.clip(np.nan_to_num(ndre1, nan=0.0, posinf=clip_val, neginf=-clip_val), -clip_val,
                                   clip_val)

    except IndexError:
        print("Error: Band index out of bounds for index calculation."); return {}
    except Exception as e:
        print(f"Error calculating indices: {e}"); traceback.print_exc(); return {}
    return indices


# --- Normalization for 16 channels ---
def normalize_16ch_per_image(image_np_16ch: np.ndarray) -> Optional[np.ndarray]:
    """Normalizes 16-channel NumPy image (C, H, W) using its own stats per channel."""
    if image_np_16ch is None or image_np_16ch.ndim != 3 or image_np_16ch.shape[0] != 16:
        print(
            f"Normalization error: Expected (16, H, W), got {image_np_16ch.shape if image_np_16ch is not None else 'None'}")
        return None
    try:
        mean = np.nanmean(image_np_16ch, axis=(1, 2), keepdims=True);
        std = np.nanstd(image_np_16ch, axis=(1, 2), keepdims=True)
        # Handle cases where std dev is zero or very small, or mean is NaN
        mean = np.nan_to_num(mean, nan=0.0)
        std = np.nan_to_num(std, nan=1.0)  # Replace NaN std with 1
        std[std < 1e-7] = 1.0  # Replace near-zero std with 1 to avoid division by zero
        normalized_image = (image_np_16ch - mean) / std
        # Clipping might be necessary depending on model and data variance
        # Consider clipping based on typical std deviations observed across dataset if needed
        # normalized_image = np.clip(normalized_image, -3.0, 3.0) # Example clip range
        if np.isnan(normalized_image).any() or np.isinf(normalized_image).any():
            print("Warning: NaN/Inf detected AFTER normalization. Replacing with 0.")
            normalized_image = np.nan_to_num(normalized_image, nan=0.0, posinf=0.0, neginf=0.0)
        return normalized_image.astype(np.float32)  # Ensure float32 output
    except Exception as e:
        print(f"Error during 16ch normalization: {e}")
        traceback.print_exc()
        return None


# --- Dataset Class (Indices BEFORE Norm) ---
class Sentinel2Dataset(Dataset):
    """ Dataset: Loads TIF, calculates indices from 12 raw bands, stacks to 16ch, then normalizes. """

    def __init__(self, paths_labels: List[Tuple[str, int]], transform: Optional[Callable] = None):
        self.paths_labels = [(p, l) for p, l in paths_labels if os.path.exists(p)]
        if len(self.paths_labels) != len(paths_labels): print(
            f"Warning: Filtered out {len(paths_labels) - len(self.paths_labels)} non-existent paths.")
        self.transform = transform
        self.output_channels = CONFIG["model"]["input_channels"]  # Should be 16
        print(
            f"Initialized Sentinel2Dataset (Indices BEFORE Norm) with {len(self.paths_labels)} samples. Output channels: {self.output_channels}")

    def __len__(self):
        return len(self.paths_labels)

    def __getitem__(self, idx):
        image_path, label = self.paths_labels[idx]
        try:
            # 1. Load TIF and select 12 bands (raw-ish float32 values)
            image_np_12 = load_sentinel2_image(image_path)
            if image_np_12 is None: return None  # Skip if loading failed

            # 2. Calculate Indices from UN-NORMALIZED 12 bands
            indices_dict = calculate_indices_from_raw(image_np_12)
            if not indices_dict: return None  # Skip if index calculation failed

            # 3. Stack 12 bands and 4 indices
            indices_list = [indices_dict['NDVI'], indices_dict['NDWI'], indices_dict['NDBI'], indices_dict['NDRE1']]
            # Ensure indices have the same H, W as bands before stacking
            h, w = image_np_12.shape[1], image_np_12.shape[2]
            indices_arrays = [idx_arr[np.newaxis, :h, :w] for idx_arr in
                              indices_list]  # Add channel dim and ensure size match

            try:
                image_np_16 = np.concatenate([image_np_12] + indices_arrays, axis=0)  # (16, H, W)
            except ValueError as e:
                print(
                    f"Error concatenating bands and indices for {image_path}: {e}. Shapes: Bands {image_np_12.shape}, Indices {[a.shape for a in indices_arrays]}")
                return None

            # Check channel count before normalization
            if image_np_16.shape[0] != self.output_channels:
                print(
                    f"Error: Stacked image channels ({image_np_16.shape[0]}) != expected ({self.output_channels}) for {image_path}. Skipping.")
                return None

            # 4. Normalize the 16 channels per image
            image_np_norm_16 = normalize_16ch_per_image(image_np_16)
            if image_np_norm_16 is None: return None  # Skip if normalization failed

            # 5. Convert to tensor
            image_tensor = torch.from_numpy(image_np_norm_16).float()  # Already float32

            # 6. Apply augmentations/transformations (resize, flips, erase, etc.)
            # NOTE: Resize should be applied here, before model expects fixed size
            if self.transform:
                image_tensor = self.transform(image_tensor)

            # Ensure tensor shape is as expected after transforms ( C, H, W )
            if image_tensor.shape[0] != self.output_channels:
                print(
                    f"Error: Tensor channels ({image_tensor.shape[0]}) != expected ({self.output_channels}) after transforms for {image_path}. Skipping.")
                return None
            if image_tensor.ndim != 3:
                print(
                    f"Error: Tensor dimensions ({image_tensor.ndim}) != 3 after transforms for {image_path}. Skipping.")
                return None

            label_tensor = torch.tensor(label, dtype=torch.long)
            return image_tensor, label_tensor, image_path

        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            traceback.print_exc()
            return None


# --- Data Transforms ---
# ResNet typically uses 224x224, update IMG_SIZE
IMG_SIZE = CONFIG["data"]["image_size"]  # Should be 224 now
# Note: Pre-trained models often have their own normalization constants (e.g., ImageNet stats)
# However, since we have 16 channels (not 3) and potentially different value ranges (L1C + indices),
# using per-image normalization or dataset-specific stats for the 16 channels is more appropriate here.
# We are currently using per-image normalization within the Dataset class.
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), antialias=True),  # Resize to model input size
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    # Optional: Add more augmentations like RandomRotation, ColorJitter (if applicable to specific channels)
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.3),  # Adjust sigma if needed
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0, inplace=False),  # Adjust params
])
val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), antialias=True),  # Resize to model input size
])

# --- Create Datasets ---
print("Scanning training directory and creating dataset splits...")
full_dataset_paths_labels = []
class_to_idx_map = {}
class_names = []
idx_counter = 0
train_root_dir = CONFIG["data"]["train_dir"]
if not train_root_dir.exists(): raise FileNotFoundError(f"Training directory not found: {train_root_dir}")
for class_folder in sorted(train_root_dir.iterdir()):
    if class_folder.is_dir() and not class_folder.name.startswith('.'):
        class_name = class_folder.name
        if class_name not in class_to_idx_map: class_to_idx_map[class_name] = idx_counter; class_names.append(
            class_name); idx_counter += 1
        class_idx = class_to_idx_map[class_name]
        for filepath in class_folder.glob('*.tif'): full_dataset_paths_labels.append((str(filepath), class_idx))

# Determine num_classes and update config
num_classes = len(class_names)
if num_classes == 0: raise FileNotFoundError(f"No valid class folders containing .tif files found in {train_root_dir}")
CONFIG["model"]["num_classes"] = num_classes;
CONFIG["model"]["class_names"] = class_names
print(f"Found {len(full_dataset_paths_labels)} training image paths in {num_classes} classes: {class_names}")
print(f"Class mapping: {class_to_idx_map}")

# Split data
try:
    train_info, val_info = train_test_split(full_dataset_paths_labels, train_size=CONFIG["data"]["train_ratio"],
                                            random_state=CONFIG["train"]["seed"],
                                            stratify=[label for _, label in full_dataset_paths_labels])
except ValueError as e:
    print(f"Warning: Stratified split failed ({e}). Falling back to non-stratified split.")
    train_info, val_info = train_test_split(full_dataset_paths_labels, train_size=CONFIG["data"]["train_ratio"],
                                            random_state=CONFIG["train"]["seed"])

print("Creating Sentinel2Dataset instances (Indices BEFORE Norm)...")
train_dataset = Sentinel2Dataset(train_info, transform=train_transforms)
val_tif_dataset = Sentinel2Dataset(val_info, transform=val_transforms)  # Renamed for clarity


# --- Create DataLoaders ---
def collate_fn(batch):  # Same collate_fn should work
    batch = [item for item in batch if item is not None]
    if not batch: return None, None, None
    try:
        images = torch.stack([item[0] for item in batch]);
        labels = torch.stack([item[1] for item in batch]);
        paths = [item[2] for item in batch]
        return images, labels, paths
    except Exception as e:
        print(f"Error in collate_fn: {e}. Skipping batch."); return None, None, None


persistent_workers = CONFIG["data"]["num_workers"] > 0
train_loader = DataLoader(train_dataset, batch_size=CONFIG["data"]["batch_size"], shuffle=True,
                          num_workers=CONFIG["data"]["num_workers"], pin_memory=True, collate_fn=collate_fn,
                          drop_last=True, persistent_workers=persistent_workers)
val_loader = DataLoader(val_tif_dataset, batch_size=CONFIG["data"]["batch_size"] * 2, shuffle=False,
                        num_workers=CONFIG["data"]["num_workers"], pin_memory=True, collate_fn=collate_fn,
                        persistent_workers=persistent_workers)
print("Train and Validation DataLoaders created.")


# --- Model Definition (Using Pre-trained ResNet50) ---

def adapt_resnet_for_multichannel(model_name="resnet50", pretrained=True, num_classes=10, input_channels=16):
    """Loads a pretrained ResNet, adapts first conv layer for N channels, replaces final FC layer."""
    print(f"Loading {'pretrained' if pretrained else 'random weights'} {model_name}...")
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.get_model(model_name, weights=weights)

    # --- Adapt the first convolutional layer ---
    # Get the original conv1 weights (typically 3 input channels)
    original_conv1 = model.conv1
    original_weights = original_conv1.weight.data  # Shape: (out_channels, 3, kernel_size, kernel_size)

    # Create a new conv layer with the desired input channels
    new_conv1 = nn.Conv2d(input_channels, original_conv1.out_channels,
                          kernel_size=original_conv1.kernel_size,
                          stride=original_conv1.stride,
                          padding=original_conv1.padding,
                          bias=original_conv1.bias is not None)

    # Initialize new weights - simple averaging/copying strategy
    # Average the original 3-channel weights and replicate across the new channels
    avg_weights = torch.mean(original_weights, dim=1, keepdim=True)  # (out_channels, 1, k, k)
    new_weights = avg_weights.repeat(1, input_channels, 1, 1)  # (out_channels, input_channels, k, k)

    # Alternatively: Initialize with Kaiming Normal (might be better if not using pretrained weights)
    # if not pretrained:
    #    nn.init.kaiming_normal_(new_conv1.weight, mode='fan_out', nonlinearity='relu')
    # else: # Assign the adapted pretrained weights
    new_conv1.weight.data = new_weights

    if new_conv1.bias is not None:
        new_conv1.bias.data = original_conv1.bias.data

    # Replace the original conv1 layer
    model.conv1 = new_conv1
    print(f"Adapted model.conv1 to accept {input_channels} input channels.")

    # --- Replace the final fully connected layer ---
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    nn.init.xavier_uniform_(model.fc.weight)  # Initialize the new head
    if model.fc.bias is not None:
        nn.init.zeros_(model.fc.bias)
    print(f"Replaced model.fc layer for {num_classes} output classes.")

    return model


# --- MixUp and CutMix Implementations (Unchanged) ---
def mixup_data(x, y, alpha=1.0, device='cuda'):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0];
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :];
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def rand_bbox(size, lam):
    W = size[2];
    H = size[3];
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat);
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W);
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W);
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W);
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def cutmix_data(x, y, alpha=1.0, device='cuda'):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0];
    index = torch.randperm(batch_size).to(device)
    y_a, y_b = y, y[index];
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]  # Corrected slicing
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, y_a, y_b, lam


# --- Training/Validation Epoch Helper (Mostly Unchanged, add backbone freezing) ---
def run_epoch(model, loader, criterion, optimizer, scaler, scheduler, device, is_training, epoch_num, num_epochs_total,
              config, is_cosine_scheduler=False):
    freeze_backbone_epochs = config['train'].get("freeze_backbone_epochs", 0)

    if is_training:
        # Handle backbone freezing/unfreezing
        if freeze_backbone_epochs > 0:
            if epoch_num <= freeze_backbone_epochs:
                print(f"Epoch {epoch_num}/{num_epochs_total}: Backbone frozen.")
                for name, param in model.named_parameters():
                    if not name.startswith('fc.'):  # Freeze everything except the final layer
                        param.requires_grad = False
            elif epoch_num == freeze_backbone_epochs + 1:
                print(f"Epoch {epoch_num}/{num_epochs_total}: Unfreezing backbone.")
                for param in model.parameters():
                    param.requires_grad = True
            # Re-create optimizer if parameters changed requires_grad status
            # Note: This is complex with AdamW state. Simpler to just ensure all params are included initially.
            # A better approach might be to use different param groups in the optimizer.
            # For now, we assume the optimizer includes all params from the start.

        model.train()
        print(
            f'---> Starting Training Epoch {epoch_num}/{num_epochs_total} | LR: {optimizer.param_groups[0]["lr"]:.4e}')
    else:
        model.eval()
        print(f'---> Starting Validation Epoch {epoch_num}/{num_epochs_total}')

    running_loss = 0.0;
    correct_predictions = 0;
    total_samples = 0;
    start_time = time.time()
    # Ensure label smoothing is correctly applied in the base criterion
    base_criterion = nn.CrossEntropyLoss(label_smoothing=config['train']['label_smoothing'], reduction='mean')
    progress_bar = tqdm(loader, desc=f"Epoch {epoch_num} {'Train' if is_training else 'Val'}", leave=False)
    gradient_accumulation_steps = config['train']['gradient_accumulation_steps']
    gradient_clip_norm = config['train'].get("gradient_clip_norm", None)
    amp_enabled = scaler is not None and config['amp_enabled']  # Check config flag too

    use_mixup = config['train']['use_mixup'] and is_training;
    use_cutmix = config['train']['use_cutmix'] and is_training
    mixup_alpha = config['train']['mixup_alpha'];
    cutmix_alpha = config['train']['cutmix_alpha'];
    mixup_prob = config['train']['mixup_prob']

    # Ensure optimizer is using correct parameters (especially if freezing changed requires_grad)
    # optimizer.param_groups[0]['params'] = [p for p in model.parameters() if p.requires_grad] # Could reset state

    with torch.set_grad_enabled(is_training):
        for batch_idx, batch_data in enumerate(progress_bar):
            if batch_data is None or batch_data[0] is None: continue
            inputs, targets, _ = batch_data
            inputs, targets_orig = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

            apply_mixup_cutmix = (use_mixup or use_cutmix) and random.random() < mixup_prob
            if apply_mixup_cutmix:
                # Randomly choose between Mixup and Cutmix if both are enabled
                use_this_mixup = use_mixup
                if use_mixup and use_cutmix: use_this_mixup = (random.random() < 0.5)

                if use_this_mixup:
                    inputs, targets_a, targets_b, lam = mixup_data(inputs, targets_orig, mixup_alpha, device)
                else:
                    inputs, targets_a, targets_b, lam = cutmix_data(inputs, targets_orig, cutmix_alpha, device)
            else:  # Ensure variables exist even if not used
                apply_mixup_cutmix = False
                targets_a, targets_b, lam = None, None, None  # Avoid potential UnboundLocalError

            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                outputs = model(inputs)
                if apply_mixup_cutmix:
                    loss = mixup_criterion(base_criterion, outputs, targets_a, targets_b, lam)
                else:
                    # Use the main criterion (which might be base_criterion or wrapped)
                    loss = criterion(outputs, targets_orig)

                if is_training and gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

            if is_training:
                if amp_enabled:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    if amp_enabled: scaler.unscale_(optimizer)
                    if gradient_clip_norm: torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                                          max_norm=gradient_clip_norm)

                    if amp_enabled:
                        scaler.step(optimizer); scaler.update()
                    else:
                        optimizer.step()

                    optimizer.zero_grad(set_to_none=True)

                    # Step scheduler after optimizer step if CosineAnnealingLR per step
                    if is_cosine_scheduler and scheduler is not None:
                        scheduler.step()  # Step per optimizer step for CosineAnnealingLR with T_max=total_steps

                if torch.isnan(loss) or torch.isinf(loss): print(
                    f"\nWARNING: NaN/Inf loss detected E{epoch_num} B{batch_idx + 1}. Loss: {loss.item()}. Skipping gradient step."); optimizer.zero_grad(
                    set_to_none=True); continue

            batch_loss = loss.item() * gradient_accumulation_steps if is_training and gradient_accumulation_steps > 1 else loss.item()
            running_loss += batch_loss * inputs.size(0)  # Use original batch size before accumulation scaling

            # Accuracy Calculation (always based on original targets)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += targets_orig.size(0)
            correct_predictions += (predicted == targets_orig).sum().item()
            current_acc = correct_predictions / total_samples if total_samples > 0 else 0

            progress_bar.set_postfix(loss=f"{batch_loss:.4f}", acc=f"{current_acc:.3f}")

    epoch_duration = time.time() - start_time
    if total_samples == 0: print(f"Warning: No valid samples processed in epoch {epoch_num}."); return 0.0, 0.0
    epoch_loss = running_loss / total_samples;
    epoch_acc = correct_predictions / total_samples
    mode_str = "Training" if is_training else "Validation"
    print(
        f'\n    Epoch {epoch_num} {mode_str} Summary: Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f} | Duration: {epoch_duration:.2f}s')
    return epoch_loss, epoch_acc


# --- Main Execution Logic ---
if __name__ == '__main__':
    # Ensure output directory exists
    base_save_path = CONFIG["model"]["base_save_path"]
    base_save_path.mkdir(parents=True, exist_ok=True)
    best_model_path = base_save_path / f"{CONFIG['model']['name']}_best.pth"  # Use updated name

    # Ensure num_classes is set
    if CONFIG["model"]["num_classes"] is None:
        raise ValueError("Number of classes not determined from dataset scan.")

    # --- Instantiate the adapted pre-trained model ---
    model = adapt_resnet_for_multichannel(
        model_name="resnet50",
        pretrained=CONFIG["model"]["pretrained"],
        num_classes=CONFIG["model"]["num_classes"],
        input_channels=CONFIG["model"]["input_channels"]  # Should be 16
    )
    model.to(DEVICE)

    # --- Define Loss and Optimizer ---
    criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG["train"]["label_smoothing"])
    print(f"Base Loss function: CrossEntropyLoss (Label Smoothing={CONFIG['train']['label_smoothing']})")

    # Consider different learning rates for backbone and head later if needed
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["train"]["lr"],
                                  weight_decay=CONFIG["train"]["weight_decay"])
    print(f"Optimizer: AdamW (LR={CONFIG['train']['lr']}, WD={CONFIG['train']['weight_decay']})")

    # --- Setup Scheduler ---
    num_epochs = CONFIG["train"]["epochs"]
    gradient_accumulation_steps = CONFIG["train"]["gradient_accumulation_steps"]
    gradient_clip_norm = CONFIG["train"].get("gradient_clip_norm", None)
    if gradient_clip_norm: print(f"Using gradient clipping with max_norm={gradient_clip_norm}")

    # Calculate total steps based on loader size AFTER accumulation
    steps_per_epoch = len(train_loader) // gradient_accumulation_steps
    if steps_per_epoch == 0:  # Handle small datasets/large accumulation
        steps_per_epoch = 1
        print(
            f"Warning: steps_per_epoch is 0 (len(train_loader)={len(train_loader)}, accum={gradient_accumulation_steps}). Setting to 1.")

    # If using CosineAnnealingLR, T_max should be total steps if stepping per batch, or total epochs if stepping per epoch
    # Current run_epoch steps per batch update, so T_max should be total steps
    t_max_steps = steps_per_epoch * num_epochs

    if CONFIG["train"]["scheduler"] == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max_steps,
                                                               eta_min=CONFIG["train"]["eta_min"])
        is_cosine_scheduler = True
        print(f"Using CosineAnnealingLR scheduler (T_max_steps={t_max_steps}, eta_min={CONFIG['train']['eta_min']})")
    else:  # Example: ReduceLROnPlateau (steps per epoch)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=7,
                                                               verbose=True)
        is_cosine_scheduler = False  # Important flag for run_epoch
        print("Using ReduceLROnPlateau scheduler.")

    # --- Training Loop Setup ---
    patience = CONFIG["train"]["patience"]
    checkpoint_interval = CONFIG["train"]["checkpoint_interval"]
    print(f"Saving checkpoints every {checkpoint_interval} epochs.")
    overall_best_val_loss = float('inf');
    best_model_state_dict = None;
    total_epochs_run = 0
    # Enable scaler only if cuda is used AND amp is enabled in config
    scaler = GradScaler(enabled=CONFIG["amp_enabled"] and DEVICE.type == 'cuda')
    if scaler.is_enabled():
        print("Using Automatic Mixed Precision (AMP).")
    else:
        print("AMP not enabled (either device is not CUDA or amp_enabled=False).")

    print(f"\n--- Starting Training for {num_epochs} epochs ---")
    epochs_without_improvement = 0
    for epoch in range(num_epochs):
        epoch_num = total_epochs_run + 1

        # --- Training Epoch ---
        train_loss, train_accuracy = run_epoch(model, train_loader, criterion, optimizer, scaler, scheduler, DEVICE,
                                               True,
                                               epoch_num, num_epochs, CONFIG, is_cosine_scheduler)

        # --- Validation Epoch ---
        avg_val_loss, val_accuracy = run_epoch(model, val_loader, criterion, None, scaler, None, DEVICE, False,
                                               epoch_num, num_epochs, CONFIG,
                                               False)  # Scheduler not stepped in validation

        # --- Scheduler Step (if epoch-based) ---
        if not is_cosine_scheduler and scheduler is not None:
            scheduler.step(avg_val_loss)  # ReduceLROnPlateau steps on metric

        current_lr = optimizer.param_groups[0]['lr']
        print(
            f'End of Epoch {epoch_num} - Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.4f} | Current LR: {current_lr:.6e}')

        # --- Checkpointing Logic ---
        is_best = avg_val_loss < overall_best_val_loss
        if is_best:
            overall_best_val_loss = avg_val_loss;
            epochs_without_improvement = 0
            try:
                # Save the state dict of the model
                best_model_state_dict = model.state_dict()
                torch.save(best_model_state_dict, best_model_path)
                print(f'---> Validation Loss Improved. Best model state saved to {best_model_path}')
            except Exception as e:
                print(f"Error saving best model state_dict: {e}")
        else:
            epochs_without_improvement += 1;
            print(f'---> Val loss did not improve for {epochs_without_improvement} epochs.')

        # Periodic checkpoint saving
        if checkpoint_interval > 0 and epoch_num % checkpoint_interval == 0:
            periodic_path = base_save_path / f"{CONFIG['model']['name']}_epoch_{epoch_num}.pth"
            try:
                torch.save(model.state_dict(), periodic_path);
                print(f'---> Periodic checkpoint saved to {periodic_path}')
            except Exception as e:
                print(f"Error saving periodic checkpoint: {e}")

        # Early stopping
        if epochs_without_improvement >= patience:
            print(f'Early stopping triggered at epoch {epoch_num} after {patience} epochs without improvement.');
            break
        # --- End Checkpointing ---
        total_epochs_run = epoch_num

    print("\n--- Training Complete ---")
    if overall_best_val_loss != float('inf'):
        print(f'Best model based on validation loss saved to {best_model_path} (Val Loss: {overall_best_val_loss:.4f})')
    else:
        print("Training completed, but no improvement in validation loss was observed.")
    print(f"Periodic checkpoints saved every {checkpoint_interval} epochs in {base_save_path}")

    # Reminder about test set evaluation
    print("\n--- Script Finished ---")
    print(
        "IMPORTANT: Remember to implement logic to load the .npy test files, apply the same preprocessing (indices->stack->norm->transform), run inference with the best checkpoint, and format the output for Kaggle submission.")