import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models
import rasterio # Keep rasterio for TIF loading
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

# --- Configuration for Training from Scratch ---
CONFIG = {
    "model": {
        "name": "ResNet50_16ch_EuroSAT_Scratch_v1", # Name reflects training from scratch
        "pretrained": False, # IMPORTANT: Set to False for Track 1
        "input_channels": 16, # 12 bands + 4 indices
        "base_save_path": Path('./outputs/resnet50_16ch_eurosat_scratch_v1'), # New path
        "num_classes": None, # To be determined from data
        "class_names": None, # To be determined from data
    },
    "data": {
        "train_dir": Path('./ds/images/remote_sensing/otherDatasets/sentinel_2/tif'),
        "image_size": 224, # ResNet standard size
        "batch_size": 32, # Adjust based on GPU memory
        "num_workers": 8,
        "train_ratio": 0.9,
    },
    "train": {
        "seed": 1337,
        "epochs": 150, # <<< Increased further
        "lr": 5e-4, # Keep LR same as the one that reached 45% initially
        "warmup_epochs": 5,
        "optimizer": "AdamW",
        "weight_decay": 1e-3, # <<< Increased weight decay
        "scheduler": "CosineAnnealingLR",
        "T_max_epochs": 150, # <<< Match increased epochs
        "eta_min": 1e-6,
        "patience": 20, # <<< Allow more patience for longer training
        "label_smoothing": 0.1,
        "gradient_accumulation_steps": 2,
        "gradient_clip_norm": 1.0,
        "use_mixup": True,
        "mixup_alpha": 0.4,
        "use_cutmix": False,
        "cutmix_alpha": 1.0,
        "mixup_prob": 1.0,
        "checkpoint_interval": 15, # <<< Save less often
    },
    "device": DEVICE.type,
    "amp_enabled": USE_CUDA,
}

# --- Seed Setting ---
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to {seed}")
set_seed(CONFIG["train"]["seed"])

# --- Data Loading and Preprocessing (Indices BEFORE Norm) ---
TARGET_BANDS_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12] # Indices for B1..B9, B11, B12, B8A

# --- Band Mapping for the 12 Channels (Reflects Standard Order) ---
# Use this map in BOTH training and evaluation scripts
BAND_MAP_12 = {
    "B1_Coastal": 0, "B2_Blue": 1, "B3_Green": 2, "B4_Red": 3,
    "B5_RE1": 4, "B6_RE2": 5, "B7_RE3": 6, "B8_NIR": 7,
    "B8A_NIR2": 8, "B9_WV": 9, "B11_SWIR1": 10, "B12_SWIR2": 11
}
print(f"Using BAND_MAP_12 reflecting standard order: {BAND_MAP_12}")


def load_sentinel2_image(filepath: str) -> Optional[np.ndarray]:
    """
    Loads a 13-band TIF file, selects 12 bands (B1-B9, B11, B12, B8A),
    and reorders them to the standard L2A sequence (B1-B8, B8A, B9, B11, B12).
    Returns (12, H, W) NumPy array or None.
    """
    try:
        filepath_lower = filepath.lower()
        if not filepath_lower.endswith(('.tif', '.tiff')): return None

        with rasterio.open(filepath) as src:
            if src.count != 13:
                print(f"Error: Expected 13 bands in TIF {filepath}, found {src.count}. Skipping.")
                return None
            # Assume standard TIF order: B1-B10 at indices 0-9, B11 at 10, B12 at 11, B8A at 12
            all_bands = src.read(list(range(1, 14))).astype(np.float32) # Read all 13 bands

        # Define indices corresponding to the standard L2A order B1-B8, B8A, B9, B11, B12
        # based on their position in the *assumed* 13-band TIF order
        # B1-B8: indices 0-7
        # B8A:   index 12
        # B9:    index 8
        # B11:   index 10
        # B12:   index 11
        standard_order_indices_in_tif = [0, 1, 2, 3, 4, 5, 6, 7, 12, 8, 10, 11]

        # Select and reorder the bands
        image_data_12_standard_order = all_bands[standard_order_indices_in_tif, :, :]

        if image_data_12_standard_order.shape[0] != 12:
             print(f"Error: Reordered image has unexpected channel count ({image_data_12_standard_order.shape[0]}). Skipping.")
             return None # Should not happen if logic is correct

        return image_data_12_standard_order

    except Exception as e:
        print(f"Error loading and reordering TIF image {filepath}: {e}")
        traceback.print_exc()
        return None
# --- Index Calculation (operates on UN-NORMALIZED 12 bands) ---
def calculate_indices_from_raw(image_np_12bands: np.ndarray, epsilon=1e-7) -> Dict[str, np.ndarray]:
    """Calculates NDVI, NDWI, NDBI, NDRE1 from an UN-NORMALIZED 12-band NumPy array. Clips output."""
    indices = {}; clip_val = 1.0 # Standard index range clipping
    global BAND_MAP_12
    try:
        nir = image_np_12bands[BAND_MAP_12["B8_NIR"], :, :]
        red = image_np_12bands[BAND_MAP_12["B4_Red"], :, :]
        green = image_np_12bands[BAND_MAP_12["B3_Green"], :, :]
        swir1 = image_np_12bands[BAND_MAP_12["B11_SWIR1"], :, :]
        re1 = image_np_12bands[BAND_MAP_12["B5_RE1"], :, :]

        denominator_ndvi = nir + red; ndvi = np.full_like(denominator_ndvi, 0.0, dtype=np.float32)
        valid_mask_ndvi = np.abs(denominator_ndvi) > epsilon
        ndvi[valid_mask_ndvi] = (nir[valid_mask_ndvi] - red[valid_mask_ndvi]) / denominator_ndvi[valid_mask_ndvi]
        indices['NDVI'] = np.clip(np.nan_to_num(ndvi, nan=0.0, posinf=clip_val, neginf=-clip_val), -clip_val, clip_val)

        denominator_ndwi = green + nir; ndwi = np.full_like(denominator_ndwi, 0.0, dtype=np.float32)
        valid_mask_ndwi = np.abs(denominator_ndwi) > epsilon
        ndwi[valid_mask_ndwi] = (green[valid_mask_ndwi] - nir[valid_mask_ndwi]) / denominator_ndwi[valid_mask_ndwi]
        indices['NDWI'] = np.clip(np.nan_to_num(ndwi, nan=0.0, posinf=clip_val, neginf=-clip_val), -clip_val, clip_val)

        denominator_ndbi = swir1 + nir; ndbi = np.full_like(denominator_ndbi, 0.0, dtype=np.float32)
        valid_mask_ndbi = np.abs(denominator_ndbi) > epsilon
        ndbi[valid_mask_ndbi] = (swir1[valid_mask_ndbi] - nir[valid_mask_ndbi]) / denominator_ndbi[valid_mask_ndbi]
        indices['NDBI'] = np.clip(np.nan_to_num(ndbi, nan=0.0, posinf=clip_val, neginf=-clip_val), -clip_val, clip_val)

        denominator_ndre1 = nir + re1; ndre1 = np.full_like(denominator_ndre1, 0.0, dtype=np.float32)
        valid_mask_ndre1 = np.abs(denominator_ndre1) > epsilon
        ndre1[valid_mask_ndre1] = (nir[valid_mask_ndre1] - re1[valid_mask_ndre1]) / denominator_ndre1[valid_mask_ndre1]
        indices['NDRE1'] = np.clip(np.nan_to_num(ndre1, nan=0.0, posinf=clip_val, neginf=-clip_val), -clip_val, clip_val)
    except IndexError: print("Error: Band index out of bounds for index calculation."); return {}
    except Exception as e: print(f"Error calculating indices: {e}"); traceback.print_exc(); return {}
    return indices

# --- Normalization for 16 channels ---
def normalize_16ch_per_image(image_np_16ch: np.ndarray) -> Optional[np.ndarray]:
    """Normalizes 16-channel NumPy image (C, H, W) using its own stats per channel."""
    if image_np_16ch is None or image_np_16ch.ndim != 3 or image_np_16ch.shape[0] != 16:
        print(f"Normalization error: Expected (16, H, W), got {image_np_16ch.shape if image_np_16ch is not None else 'None'}")
        return None
    try:
        mean = np.nanmean(image_np_16ch, axis=(1, 2), keepdims=True);
        std = np.nanstd(image_np_16ch, axis=(1, 2), keepdims=True)
        mean = np.nan_to_num(mean, nan=0.0)
        std = np.nan_to_num(std, nan=1.0)
        std[std < 1e-7] = 1.0
        normalized_image = (image_np_16ch - mean) / std
        if np.isnan(normalized_image).any() or np.isinf(normalized_image).any():
            print("Warning: NaN/Inf detected AFTER normalization. Replacing with 0.")
            normalized_image = np.nan_to_num(normalized_image, nan=0.0, posinf=0.0, neginf=0.0)
        return normalized_image.astype(np.float32)
    except Exception as e:
        print(f"Error during 16ch normalization: {e}")
        traceback.print_exc()
        return None


# --- Dataset Class (Indices BEFORE Norm) ---
class Sentinel2Dataset(Dataset):
    """ Dataset: Loads TIF, calculates indices from 12 raw bands, stacks to 16ch, then normalizes. """
    def __init__(self, paths_labels: List[Tuple[str, int]], transform: Optional[Callable] = None):
        self.paths_labels = [(p, l) for p, l in paths_labels if os.path.exists(p)]
        if len(self.paths_labels) != len(paths_labels): print(f"Warning: Filtered out {len(paths_labels) - len(self.paths_labels)} non-existent paths.")
        self.transform = transform
        self.output_channels = CONFIG["model"]["input_channels"] # Should be 16
        print(f"Initialized Sentinel2Dataset (Indices BEFORE Norm) with {len(self.paths_labels)} samples. Output channels: {self.output_channels}")

    def __len__(self): return len(self.paths_labels)

    def __getitem__(self, idx):
        image_path, label = self.paths_labels[idx]
        try:
            image_np_12 = load_sentinel2_image(image_path)
            if image_np_12 is None: return None
            indices_dict = calculate_indices_from_raw(image_np_12)
            if not indices_dict: return None
            indices_list = [indices_dict['NDVI'], indices_dict['NDWI'], indices_dict['NDBI'], indices_dict['NDRE1']]
            h, w = image_np_12.shape[1], image_np_12.shape[2]
            indices_arrays = [idx_arr[np.newaxis, :h, :w] for idx_arr in indices_list]
            try:
                image_np_16 = np.concatenate([image_np_12] + indices_arrays, axis=0)
            except ValueError as e:
                 print(f"Error concatenating bands and indices for {image_path}: {e}. Shapes: Bands {image_np_12.shape}, Indices {[a.shape for a in indices_arrays]}")
                 return None
            if image_np_16.shape[0] != self.output_channels: return None # Should already be checked
            image_np_norm_16 = normalize_16ch_per_image(image_np_16)
            if image_np_norm_16 is None: return None
            image_tensor = torch.from_numpy(image_np_norm_16).float()
            if self.transform: image_tensor = self.transform(image_tensor)
            if image_tensor.shape[0] != self.output_channels or image_tensor.ndim != 3: return None # Final check

            label_tensor = torch.tensor(label, dtype=torch.long)
            return image_tensor, label_tensor, image_path
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            traceback.print_exc()
            return None

# --- Data Transforms ---
IMG_SIZE = CONFIG["data"]["image_size"] # Should be 224
# Simple transforms are often better when starting training from scratch
# Avoid RandomCrop initially based on previous feedback
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), antialias=True), # Resize to model input size
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.3),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0, inplace=False),
])
val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), antialias=True), # Resize to model input size
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
         if class_name not in class_to_idx_map: class_to_idx_map[class_name] = idx_counter; class_names.append(class_name); idx_counter += 1
         class_idx = class_to_idx_map[class_name]
         for filepath in class_folder.glob('*.tif'): full_dataset_paths_labels.append((str(filepath), class_idx))

num_classes = len(class_names)
if num_classes == 0: raise FileNotFoundError(f"No valid class folders containing .tif files found in {train_root_dir}")
CONFIG["model"]["num_classes"] = num_classes; CONFIG["model"]["class_names"] = class_names
print(f"Found {len(full_dataset_paths_labels)} training image paths in {num_classes} classes: {class_names}")
print(f"Class mapping: {class_to_idx_map}")

try:
    train_info, val_info = train_test_split(full_dataset_paths_labels, train_size=CONFIG["data"]["train_ratio"], random_state=CONFIG["train"]["seed"], stratify=[label for _, label in full_dataset_paths_labels])
except ValueError as e:
     print(f"Warning: Stratified split failed ({e}). Falling back to non-stratified split.")
     train_info, val_info = train_test_split(full_dataset_paths_labels, train_size=CONFIG["data"]["train_ratio"], random_state=CONFIG["train"]["seed"])

print("Creating Sentinel2Dataset instances (Indices BEFORE Norm)...")
train_dataset = Sentinel2Dataset(train_info, transform=train_transforms)
val_tif_dataset = Sentinel2Dataset(val_info, transform=val_transforms)

# --- Create DataLoaders ---
def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch: return None, None, None
    try:
        images = torch.stack([item[0] for item in batch]); labels = torch.stack([item[1] for item in batch]); paths = [item[2] for item in batch]
        return images, labels, paths
    except Exception as e: print(f"Error in collate_fn: {e}. Skipping batch."); return None, None, None

persistent_workers = CONFIG["data"]["num_workers"] > 0
train_loader = DataLoader(train_dataset, batch_size=CONFIG["data"]["batch_size"], shuffle=True, num_workers=CONFIG["data"]["num_workers"], pin_memory=True, collate_fn=collate_fn, drop_last=True, persistent_workers=persistent_workers)
val_loader = DataLoader(val_tif_dataset, batch_size=CONFIG["data"]["batch_size"]*2, shuffle=False, num_workers=CONFIG["data"]["num_workers"], pin_memory=True, collate_fn=collate_fn, persistent_workers=persistent_workers)
print("Train and Validation DataLoaders created.")


# --- Model Definition (Using ResNet50 Architecture, Random Weights) ---
def adapt_resnet_for_multichannel(model_name="resnet50", pretrained=False, num_classes=10, input_channels=16):
    """Loads ResNet architecture, adapts first conv layer for N channels, replaces final FC layer."""
    print(f"Loading {model_name} architecture with random weights...")
    # Load architecture only, weights=None
    model = models.get_model(model_name, weights=None)

    # --- Adapt the first convolutional layer ---
    original_conv1 = model.conv1
    new_conv1 = nn.Conv2d(input_channels, original_conv1.out_channels,
                          kernel_size=original_conv1.kernel_size,
                          stride=original_conv1.stride,
                          padding=original_conv1.padding,
                          bias=original_conv1.bias is not None)
    # Initialize new conv1 weights using Kaiming Normal (good practice for ReLU/Mish)
    nn.init.kaiming_normal_(new_conv1.weight, mode='fan_out', nonlinearity='relu')
    if new_conv1.bias is not None:
        nn.init.zeros_(new_conv1.bias)
    model.conv1 = new_conv1
    print(f"Adapted model.conv1 to accept {input_channels} input channels (randomly initialized).")

    # --- Replace and initialize the final fully connected layer ---
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    nn.init.xavier_uniform_(model.fc.weight) # Initialize the new head
    if model.fc.bias is not None:
         nn.init.zeros_(model.fc.bias)
    print(f"Replaced model.fc layer for {num_classes} output classes (randomly initialized).")

    # Note: Other layers retain their default random initializations from torchvision
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
def rand_bbox(size, lam):
    W = size[2]; H = size[3]; cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat); cut_h = int(H * cut_rat)
    cx = np.random.randint(W); cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W); bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W); bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2
def cutmix_data(x, y, alpha=1.0, device='cuda'):
    if alpha > 0: lam = np.random.beta(alpha, alpha)
    else: lam = 1
    batch_size = x.size()[0]; index = torch.randperm(batch_size).to(device)
    y_a, y_b = y, y[index]; bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, y_a, y_b, lam


# --- Learning Rate Warm-up Helper ---
def get_lr(optimizer):
    # Utility to get current learning rate
    # Add check for optimizer being None
    if optimizer is None:
        return 0.0
    for param_group in optimizer.param_groups:
        return param_group['lr']

def adjust_lr_with_warmup(optimizer, initial_lr, epoch, step, steps_per_epoch, warmup_epochs, config):
    """ Adjusts learning rate with linear warm-up """
    total_warmup_steps = warmup_epochs * steps_per_epoch
    current_step = epoch * steps_per_epoch + step

    if current_step < total_warmup_steps:
        # Linear warm-up phase
        # Start from a very small LR (e.g., initial_lr / 100 or a fixed small value)
        start_lr = 1e-7 # Fixed low start LR for warm-up
        lr = start_lr + (initial_lr - start_lr) * (current_step + 1) / total_warmup_steps
    else:
        # After warm-up, use the main scheduler (which might adjust LR further)
        # This adjustment relies on the main scheduler being stepped elsewhere
        # We return None to signal that the main scheduler should handle it
        return None # Signal to use main scheduler

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr # Return the adjusted LR


# --- Training/Validation Epoch Helper (With fixes for None optimizer) ---
def run_epoch(model, loader, criterion, optimizer, scaler, scheduler, device, is_training,
              epoch_num, total_epochs, config, steps_per_epoch, is_cosine_scheduler=False):

    initial_lr = config['train']['lr']
    warmup_epochs = config['train']['warmup_epochs']

    if is_training:
        model.train()
        print(f'---> Starting Training Epoch {epoch_num}/{total_epochs} | Initial LR (post-warmup): {initial_lr:.4e}')
    else:
        model.eval()
        print(f'---> Starting Validation Epoch {epoch_num}/{total_epochs}')

    running_loss = 0.0; correct_predictions = 0; total_samples = 0; start_time = time.time()
    base_criterion = nn.CrossEntropyLoss(label_smoothing=config['train']['label_smoothing'], reduction='mean')
    progress_bar = tqdm(loader, desc=f"Epoch {epoch_num} {'Train' if is_training else 'Val'}", leave=False)
    gradient_accumulation_steps = config['train']['gradient_accumulation_steps']
    gradient_clip_norm = config['train'].get("gradient_clip_norm", None)
    amp_enabled = scaler is not None and config['amp_enabled']

    use_mixup = config['train']['use_mixup'] and is_training; use_cutmix = config['train']['use_cutmix'] and is_training
    mixup_alpha = config['train']['mixup_alpha']; cutmix_alpha = config['train']['cutmix_alpha']; mixup_prob = config['train']['mixup_prob']

    with torch.set_grad_enabled(is_training):
        for batch_idx, batch_data in enumerate(progress_bar):
            if batch_data is None or batch_data[0] is None: continue
            inputs, targets, _ = batch_data
            inputs, targets_orig = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

            # --- LR Warm-up Adjustment (Per Step) ---
            current_lr = None
            if is_training and warmup_epochs > 0 and (epoch_num-1) < warmup_epochs : # Only adjust during warm-up epochs
                 current_lr = adjust_lr_with_warmup(optimizer, initial_lr, epoch_num-1, batch_idx, steps_per_epoch, warmup_epochs, config)
            # -----------------------------------------

            apply_mixup_cutmix = (use_mixup or use_cutmix) and random.random() < mixup_prob
            if apply_mixup_cutmix:
                use_this_mixup = use_mixup
                if use_mixup and use_cutmix: use_this_mixup = (random.random() < 0.5)
                if use_this_mixup: inputs, targets_a, targets_b, lam = mixup_data(inputs, targets_orig, mixup_alpha, device)
                else: inputs, targets_a, targets_b, lam = cutmix_data(inputs, targets_orig, cutmix_alpha, device)
            else: apply_mixup_cutmix = False; targets_a, targets_b, lam = None, None, None

            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                outputs = model(inputs)
                if apply_mixup_cutmix: loss = mixup_criterion(base_criterion, outputs, targets_a, targets_b, lam)
                else: loss = criterion(outputs, targets_orig)
                if is_training and gradient_accumulation_steps > 1: loss = loss / gradient_accumulation_steps

            if is_training:
                if amp_enabled: scaler.scale(loss).backward()
                else: loss.backward()

                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    if amp_enabled: scaler.unscale_(optimizer);
                    if gradient_clip_norm: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
                    if amp_enabled: scaler.step(optimizer); scaler.update()
                    else: optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                    # --- Step Main Scheduler (AFTER warm-up phase and optimizer step) ---
                    is_after_warmup = (warmup_epochs == 0) or ((epoch_num-1) >= warmup_epochs)
                    if is_after_warmup and is_cosine_scheduler and scheduler is not None:
                         scheduler.step() # Step per optimizer step for CosineAnnealingLR
                    # ----------------------------------------------------------------------

                if torch.isnan(loss) or torch.isinf(loss): print(f"\nWARNING: NaN/Inf loss detected E{epoch_num} B{batch_idx+1}. Loss: {loss.item()}. Skipping gradient step."); optimizer.zero_grad(set_to_none=True); continue

            batch_loss = loss.item() * gradient_accumulation_steps if is_training and gradient_accumulation_steps > 1 else loss.item()
            running_loss += batch_loss * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1); total_samples += targets_orig.size(0)
            correct_predictions += (predicted == targets_orig).sum().item()
            current_acc = correct_predictions / total_samples if total_samples > 0 else 0

            # Display current LR in progress bar *only during training*
            if is_training:
                # Get the actual current LR from optimizer after potential adjustments
                display_lr = get_lr(optimizer)
                progress_bar.set_postfix(loss=f"{batch_loss:.4f}", acc=f"{current_acc:.3f}", lr=f"{display_lr:.2e}")
            else:
                # Don't display LR during validation
                progress_bar.set_postfix(loss=f"{batch_loss:.4f}", acc=f"{current_acc:.3f}")

    # --- After loop finishes ---
    epoch_duration = time.time() - start_time
    if total_samples == 0:
        print(f"Warning: No valid samples processed in epoch {epoch_num}.")
        return 0.0, 0.0 # Return zero loss/accuracy

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    mode_str = "Training" if is_training else "Validation"

    # --- FIX 2: Only get final_lr if training ---
    final_lr = None
    if is_training:
        final_lr = get_lr(optimizer)
        print(f'\n    Epoch {epoch_num} {mode_str} Summary: Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f} | Final LR: {final_lr:.4e} | Duration: {epoch_duration:.2f}s')
    else:
        # Don't print LR for validation summary
        print(f'\n    Epoch {epoch_num} {mode_str} Summary: Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f} | Duration: {epoch_duration:.2f}s')
    # --- End FIX 2 ---

    return epoch_loss, epoch_acc


# --- Main Execution Logic ---
if __name__ == '__main__':
    base_save_path = CONFIG["model"]["base_save_path"]
    base_save_path.mkdir(parents=True, exist_ok=True)
    best_model_path = base_save_path / f"{CONFIG['model']['name']}_best.pth"

    if CONFIG["model"]["num_classes"] is None: raise ValueError("Num classes not determined.")

    # --- Instantiate the adapted model (Random Weights) ---
    model = adapt_resnet_for_multichannel(
        model_name="resnet50",
        pretrained=CONFIG["model"]["pretrained"], # Should be False
        num_classes=CONFIG["model"]["num_classes"],
        input_channels=CONFIG["model"]["input_channels"]
    )
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG["train"]["label_smoothing"])
    print(f"Base Loss function: CrossEntropyLoss (Label Smoothing={CONFIG['train']['label_smoothing']})")

    optimizer = torch.optim.AdamW( model.parameters(), lr=CONFIG["train"]["lr"], weight_decay=CONFIG["train"]["weight_decay"])
    print(f"Optimizer: AdamW (Target Initial LR={CONFIG['train']['lr']}, WD={CONFIG['train']['weight_decay']})")

    # Set initial LR very low for warm-up phase; adjust_lr_with_warmup will increase it
    if CONFIG["train"]["warmup_epochs"] > 0:
        print(f"Using {CONFIG['train']['warmup_epochs']} warm-up epochs.")
        for param_group in optimizer.param_groups:
             param_group['lr'] = 1e-7 # Start warm-up from this low value

    num_epochs = CONFIG["train"]["epochs"]
    warmup_epochs = CONFIG["train"]["warmup_epochs"]
    gradient_accumulation_steps = CONFIG["train"]["gradient_accumulation_steps"]
    gradient_clip_norm = CONFIG["train"].get("gradient_clip_norm", None)
    if gradient_clip_norm: print(f"Using gradient clipping with max_norm={gradient_clip_norm}")

    steps_per_epoch = len(train_loader) // gradient_accumulation_steps
    if steps_per_epoch == 0: steps_per_epoch = 1; print(f"Warning: steps_per_epoch is 0. Setting to 1.")

    # Adjust T_max for CosineAnnealingLR to account for warm-up epochs
    total_training_steps_after_warmup = steps_per_epoch * (num_epochs - warmup_epochs)
    if total_training_steps_after_warmup <= 0:
         print("Warning: total_training_steps_after_warmup is zero or negative. Scheduler may not work as expected.")
         total_training_steps_after_warmup = 1 # Avoid zero T_max

    if CONFIG["train"]["scheduler"] == "CosineAnnealingLR":
        # Scheduler only starts *after* warm-up, base LR should be target LR
        # Create scheduler with the final target LR from config
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( optimizer, T_max=total_training_steps_after_warmup, eta_min=CONFIG["train"]["eta_min"])
        is_cosine_scheduler = True
        print(f"Using CosineAnnealingLR scheduler (T_max_steps={total_training_steps_after_warmup} after warm-up, eta_min={CONFIG['train']['eta_min']})")
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=7, verbose=True)
        is_cosine_scheduler = False
        print("Using ReduceLROnPlateau scheduler (Warm-up interaction not explicitly handled).")


    patience = CONFIG["train"]["patience"]
    checkpoint_interval = CONFIG["train"]["checkpoint_interval"]
    print(f"Saving checkpoints every {checkpoint_interval} epochs.")
    overall_best_val_loss = float('inf'); best_model_state_dict = None; total_epochs_run = 0
    scaler = GradScaler(enabled=CONFIG["amp_enabled"] and DEVICE.type == 'cuda')
    if scaler.is_enabled(): print("Using Automatic Mixed Precision (AMP).")
    else: print("AMP not enabled.")

    print(f"\n--- Starting Training from Scratch for {num_epochs} epochs ---")
    epochs_without_improvement = 0
    for epoch in range(num_epochs):
        epoch_num = total_epochs_run + 1 # Epochs are 1-based

        train_loss, train_accuracy = run_epoch( model, train_loader, criterion, optimizer, scaler, scheduler, DEVICE, True,
            epoch_num, num_epochs, CONFIG, steps_per_epoch, is_cosine_scheduler )

        avg_val_loss, val_accuracy = run_epoch( model, val_loader, criterion, None, scaler, None, DEVICE, False,
            epoch_num, num_epochs, CONFIG, steps_per_epoch, False ) # Pass steps_per_epoch even for val? Not really used there

        # --- Step epoch-based scheduler (if not Cosine and after warm-up) ---
        is_after_warmup = (warmup_epochs == 0) or (epoch_num > warmup_epochs)
        if is_after_warmup and not is_cosine_scheduler and scheduler is not None:
            scheduler.step(avg_val_loss) # ReduceLROnPlateau steps on metric

        # LR Logging moved inside run_epoch summary print

        # --- Checkpointing Logic ---
        is_best = avg_val_loss < overall_best_val_loss
        if is_best:
            overall_best_val_loss = avg_val_loss; epochs_without_improvement = 0
            try: best_model_state_dict = model.state_dict(); torch.save(best_model_state_dict, best_model_path); print(f'---> Validation Loss Improved. Best model state saved to {best_model_path}')
            except Exception as e: print(f"Error saving best model state_dict: {e}")
        else:
            epochs_without_improvement += 1; print(f'---> Val loss did not improve for {epochs_without_improvement} epochs.')

        if checkpoint_interval > 0 and epoch_num % checkpoint_interval == 0:
            periodic_path = base_save_path / f"{CONFIG['model']['name']}_epoch_{epoch_num}.pth"
            try: torch.save(model.state_dict(), periodic_path); print(f'---> Periodic checkpoint saved to {periodic_path}')
            except Exception as e: print(f"Error saving periodic checkpoint: {e}")

        if epochs_without_improvement >= patience: print(f'Early stopping triggered at epoch {epoch_num}.'); break
        # --- End Checkpointing ---
        total_epochs_run = epoch_num

    print("\n--- Training Complete ---")
    if overall_best_val_loss != float('inf'): print(f'Best model based on validation loss saved to {best_model_path} (Val Loss: {overall_best_val_loss:.4f})')
    else: print("Training completed, but no improvement in validation loss was observed.")
    print(f"Periodic checkpoints saved every {checkpoint_interval} epochs in {base_save_path}")

    print("\n--- Script Finished ---")
    print("IMPORTANT: Evaluate the best checkpoint using the evaluation script.")