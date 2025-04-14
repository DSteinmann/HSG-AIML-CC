import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
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
        "name": "CustomSentinelCNN_IdxAfterNorm_v1", # Updated name
        "input_channels": 16, # 12 bands + 4 indices
        "base_save_path": Path('./outputs/cnn_idx_after_norm_v1'), # Base path for saving checkpoints
        "num_classes": None,
        "class_names": None,
    },
    "data": {
        "train_dir": Path('./ds/images/remote_sensing/otherDatasets/sentinel_2/tif'),
        "image_size": 128,
        "batch_size": 32,
        "num_workers": 8,
        "train_ratio": 0.9,
    },
    "train": {
        "seed": 1337,
        "epochs": 50, # Keep reduced epochs
        "lr": 3e-4,
        "optimizer": "AdamW",
        "weight_decay": 3e-4, # Use WD from 40% run
        "dropout_head_1": 0.6,
        "dropout_head_2": 0.4,
        "scheduler": "CosineAnnealingLR",
        "T_max": 50, # Match reduced epochs
        "eta_min": 1e-6,
        "patience": 15,
        "label_smoothing": 0.1,
        "gradient_accumulation_steps": 2,
        "gradient_clip_norm": 1.0,
        "use_mixup": True, # Keep MixUp
        "mixup_alpha": 0.4,
        "use_cutmix": False,
        "cutmix_alpha": 1.0,
        "mixup_prob": 1.0,
        "checkpoint_interval": 5, # Keep frequent checkpoints
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

# --- Data Loading and Preprocessing ---
TARGET_BANDS_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12]
BAND_MAP = { "B1_Coastal": 0, "B2_Blue": 1, "B3_Green": 2, "B4_Red": 3, "B5_RE1": 4, "B6_RE2": 5, "B7_RE3": 6, "B8_NIR": 7, "B9_WV": 8, "B11_SWIR1": 9, "B12_SWIR2": 10, "B8A_NIR2": 11 }

def load_sentinel2_image(filepath: str) -> Optional[np.ndarray]:
    """Loads a 13-band TIF file, selects 12 specific bands, returns (12, H, W) NumPy array or None."""
    global TARGET_BANDS_INDICES
    try:
        filepath_lower = filepath.lower()
        if not filepath_lower.endswith(('.tif', '.tiff')): return None
        with rasterio.open(filepath) as src:
            if src.count == 13: all_bands = src.read(list(range(1, 14))); image_data = all_bands[TARGET_BANDS_INDICES, :, :]
            else: print(f"Error: Expected 13 bands in TIF {filepath}, found {src.count}. Skipping."); return None
        if image_data is None or image_data.shape[0] != 12: return None
        return image_data.astype(np.float32)
    except Exception as e: print(f"Error loading image {filepath}: {e}"); traceback.print_exc(); return None

# --- Normalization for 12 channels ---
def normalize_image_per_image(image_np: np.ndarray) -> Optional[np.ndarray]:
    """Normalizes 12-channel NumPy image (C, H, W) using its own stats."""
    if image_np is None or image_np.ndim != 3 or image_np.shape[0] != 12: return None
    mean = np.nanmean(image_np, axis=(1, 2), keepdims=True); std = np.nanstd(image_np, axis=(1, 2), keepdims=True)
    mean = np.nan_to_num(mean, nan=0.0); std = np.nan_to_num(std, nan=1.0)
    std_safe = std + 1e-7; normalized_image = (image_np - mean) / std_safe
    normalized_image = np.clip(normalized_image, -6.0, 6.0) # Keep clipping from previous versions
    if np.isnan(normalized_image).any() or np.isinf(normalized_image).any():
        normalized_image = np.nan_to_num(normalized_image, nan=0.0, posinf=0.0, neginf=0.0)
    return normalized_image

# --- Index Calculation (operates on NORMALIZED 12 bands) ---
def calculate_indices(image_np_12bands_normalized: np.ndarray, epsilon=1e-6) -> Dict[str, np.ndarray]:
    """Calculates NDVI, NDWI, NDBI, NDRE1 from a NORMALIZED 12-band NumPy array. Clips output."""
    indices = {}; clip_val = 10.0 # Wider clip range might be needed for indices from normalized data
    try:
        # Get bands from the already normalized 12-band input
        nir = image_np_12bands_normalized[BAND_MAP["B8_NIR"], :, :]
        red = image_np_12bands_normalized[BAND_MAP["B4_Red"], :, :]
        green = image_np_12bands_normalized[BAND_MAP["B3_Green"], :, :]
        swir1 = image_np_12bands_normalized[BAND_MAP["B11_SWIR1"], :, :]
        re1 = image_np_12bands_normalized[BAND_MAP["B5_RE1"], :, :]

        # NDVI
        denominator_ndvi = nir + red; ndvi = np.full_like(denominator_ndvi, 0.0, dtype=np.float32)
        valid_mask_ndvi = np.abs(denominator_ndvi) > epsilon
        ndvi[valid_mask_ndvi] = (nir[valid_mask_ndvi] - red[valid_mask_ndvi]) / denominator_ndvi[valid_mask_ndvi]
        indices['NDVI'] = np.clip(np.nan_to_num(ndvi, nan=0.0, posinf=clip_val, neginf=-clip_val), -clip_val, clip_val)
        # NDWI
        denominator_ndwi = green + nir; ndwi = np.full_like(denominator_ndwi, 0.0, dtype=np.float32)
        valid_mask_ndwi = np.abs(denominator_ndwi) > epsilon
        ndwi[valid_mask_ndwi] = (green[valid_mask_ndwi] - nir[valid_mask_ndwi]) / denominator_ndwi[valid_mask_ndwi]
        indices['NDWI'] = np.clip(np.nan_to_num(ndwi, nan=0.0, posinf=clip_val, neginf=-clip_val), -clip_val, clip_val)
        # NDBI
        denominator_ndbi = swir1 + nir; ndbi = np.full_like(denominator_ndbi, 0.0, dtype=np.float32)
        valid_mask_ndbi = np.abs(denominator_ndbi) > epsilon
        ndbi[valid_mask_ndbi] = (swir1[valid_mask_ndbi] - nir[valid_mask_ndbi]) / denominator_ndbi[valid_mask_ndbi]
        indices['NDBI'] = np.clip(np.nan_to_num(ndbi, nan=0.0, posinf=clip_val, neginf=-clip_val), -clip_val, clip_val)
        # NDRE1
        denominator_ndre1 = nir + re1; ndre1 = np.full_like(denominator_ndre1, 0.0, dtype=np.float32)
        valid_mask_ndre1 = np.abs(denominator_ndre1) > epsilon
        ndre1[valid_mask_ndre1] = (nir[valid_mask_ndre1] - re1[valid_mask_ndre1]) / denominator_ndre1[valid_mask_ndre1]
        indices['NDRE1'] = np.clip(np.nan_to_num(ndre1, nan=0.0, posinf=clip_val, neginf=-clip_val), -clip_val, clip_val)
    except IndexError: print("Error: Band index out of bounds for index calculation."); return {}
    except Exception as e: print(f"Error calculating indices: {e}"); traceback.print_exc(); return {}
    return indices

# --- normalize_image_per_image_16ch removed ---

# --- Dataset Class (Indices AFTER Norm) ---
class Sentinel2Dataset(Dataset):
    """ Dataset: Loads TIF, normalizes 12 bands, calculates indices, stacks to 16ch. """
    def __init__(self, paths_labels: List[Tuple[str, int]], transform: Optional[Callable] = None):
        self.paths_labels = [(p, l) for p, l in paths_labels if os.path.exists(p)]
        if len(self.paths_labels) != len(paths_labels): print(f"Warning: Filtered out {len(paths_labels) - len(self.paths_labels)} non-existent paths.")
        self.transform = transform
        self.output_channels = CONFIG["model"]["input_channels"] # Should be 16
        print(f"Initialized Sentinel2Dataset (Indices After Norm) with {len(self.paths_labels)} samples. Output channels: {self.output_channels}")

    def __len__(self): return len(self.paths_labels)

    def __getitem__(self, idx):
        image_path, label = self.paths_labels[idx]
        try:
            # 1. Load TIF and select 12 bands (raw-ish values)
            image_np_12 = load_sentinel2_image(image_path)
            if image_np_12 is None: return None

            # 2. Normalize the 12 bands per image
            image_np_norm_12 = normalize_image_per_image(image_np_12)
            if image_np_norm_12 is None: return None

            # 3. Calculate Indices from NORMALIZED bands
            indices_dict = calculate_indices(image_np_norm_12)
            if not indices_dict: return None

            # 4. Stack normalized bands and indices
            indices_list = [indices_dict['NDVI'], indices_dict['NDWI'], indices_dict['NDBI'], indices_dict['NDRE1']]
            indices_arrays = [idx[np.newaxis, :, :] for idx in indices_list]
            final_image_np = np.concatenate([image_np_norm_12] + indices_arrays, axis=0) # (16, H, W)

            # Check final channel count
            if final_image_np.shape[0] != self.output_channels:
                print(f"Error: Final image channels ({final_image_np.shape[0]}) != expected ({self.output_channels}) for {image_path}. Skipping.")
                return None

            # 5. Convert to tensor
            image_tensor = torch.from_numpy(final_image_np).float()

            # 6. Apply augmentations/transformations (flips, resize, erase, etc.)
            if self.transform:
                image_tensor = self.transform(image_tensor)

            label_tensor = torch.tensor(label, dtype=torch.long)
            return image_tensor, label_tensor, image_path

        except Exception as e:
            print(f"Error processing train/val image {image_path}: {e}")
            traceback.print_exc()
            return None

# --- Data Transforms ---
IMG_SIZE = CONFIG["data"]["image_size"]
train_transforms = transforms.Compose([
    # Apply spatial transforms first
    transforms.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    # Then intensity/noise transforms
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.4),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.20), ratio=(0.3, 3.3), value=0, inplace=False),
])
val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
])

# --- Create Datasets ---
print("Scanning training directory and creating dataset splits...")
# (Dataset scanning logic remains the same)
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
CONFIG["model"]["num_classes"] = num_classes; CONFIG["model"]["class_names"] = class_names
if num_classes == 0: raise FileNotFoundError(f"No valid class folders containing .tif files found in {train_root_dir}")
if not full_dataset_paths_labels: raise FileNotFoundError(f"No .tif files found in any class subdirectories of {train_root_dir}")
print(f"Found {len(full_dataset_paths_labels)} training image paths in {num_classes} classes: {class_names}")
print(f"Class mapping: {class_to_idx_map}")
try:
    train_info, val_info = train_test_split(full_dataset_paths_labels, train_size=CONFIG["data"]["train_ratio"], random_state=CONFIG["train"]["seed"], stratify=[label for _, label in full_dataset_paths_labels])
except ValueError as e:
     print(f"Warning: Stratified split failed ({e}). Falling back to non-stratified split.")
     train_info, val_info = train_test_split(full_dataset_paths_labels, train_size=CONFIG["data"]["train_ratio"], random_state=CONFIG["train"]["seed"])
print("Creating Sentinel2Dataset instances for training and validation...")
train_dataset = Sentinel2Dataset(train_info, transform=train_transforms)
val_tif_dataset = Sentinel2Dataset(val_info, transform=val_transforms)

# --- Create DataLoaders ---
def collate_fn(batch): # Same collate_fn
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


# --- Custom Model Definition (ResNet-like Blocks - Unchanged Architecture) ---
class Mish(nn.Module):
    def forward(self, x): return x * torch.tanh(F.softplus(x))
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__(); self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False), Mish(), nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False), nn.Sigmoid())
    def forward(self, x): return x * self.excitation(self.squeeze(x))
class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_se=True):
        super().__init__(); self.stride = stride
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels); self.activation = Mish()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels); self.se = SEBlock(out_channels) if use_se else nn.Identity()
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(out_channels))
    def forward(self, x):
        residual = self.shortcut(x); out = self.conv1(x); out = self.bn1(out); out = self.activation(out)
        out = self.conv2(out); out = self.bn2(out); out = self.se(out); out += residual; out = self.activation(out)
        return out
class CustomSentinelCNN_v2(nn.Module): # Keeping v2 name
    def __init__(self, num_classes: int, input_channels: int = CONFIG["model"]["input_channels"]):
        super().__init__()
        print(f"Creating CustomSentinelCNN_v2 for {input_channels} channels and {num_classes} classes.")
        if input_channels <= 0: raise ValueError("input_channels must be positive.")
        self.stem = nn.Sequential( nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(64), Mish(), nn.MaxPool2d(kernel_size=2, stride=2) )
        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, num_blocks=2, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1); self.max_pool = nn.AdaptiveMaxPool2d(1); pooled_features = 512
        dropout1 = CONFIG["train"]["dropout_head_1"]
        dropout2 = CONFIG["train"]["dropout_head_2"]
        self.head = nn.Sequential( nn.Flatten(), nn.BatchNorm1d(pooled_features * 2), nn.Dropout(dropout1),
            nn.Linear(pooled_features * 2, 256), Mish(), nn.BatchNorm1d(256), nn.Dropout(dropout2),
            nn.Linear(256, num_classes) )
        self.apply(self._initialize_weights)
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []; layers.append(ResidualConvBlock(in_channels, out_channels, stride=stride))
        for _ in range(1, num_blocks): layers.append(ResidualConvBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    def _initialize_weights(self, m):
        if isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)): nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias) if m.bias is not None else None
    def forward(self, x):
        x = self.stem(x); x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        avg_p = self.avg_pool(x); max_p = self.max_pool(x); x = torch.cat((avg_p, max_p), dim=1); x = self.head(x)
        return x

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
    x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2] # Corrected slicing
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, y_a, y_b, lam


# --- Training/Validation Epoch Helper (Unchanged) ---
def run_epoch(model, loader, criterion, optimizer, scaler, scheduler, device, is_training, epoch_num, num_epochs_total, config, is_cosine_scheduler=False):
    if is_training: model.train(); print(f'---> Starting Training Epoch {epoch_num}/{num_epochs_total} | LR: {optimizer.param_groups[0]["lr"]:.4e}')
    else: model.eval(); print(f'---> Starting Validation Epoch {epoch_num}/{num_epochs_total}')
    running_loss = 0.0; correct_predictions = 0; total_samples = 0; start_time = time.time()
    base_criterion = nn.CrossEntropyLoss(label_smoothing=config['train']['label_smoothing'], reduction='mean')
    progress_bar = tqdm(loader, desc=f"Epoch {epoch_num} {'Train' if is_training else 'Val'}", leave=False)
    gradient_accumulation_steps = config['train']['gradient_accumulation_steps']
    gradient_clip_norm = config['train'].get("gradient_clip_norm", None)
    amp_enabled = scaler is not None
    use_mixup = config['train']['use_mixup'] and is_training; use_cutmix = config['train']['use_cutmix'] and is_training
    mixup_alpha = config['train']['mixup_alpha']; cutmix_alpha = config['train']['cutmix_alpha']; mixup_prob = config['train']['mixup_prob']
    with torch.set_grad_enabled(is_training):
        for batch_idx, batch_data in enumerate(progress_bar):
            if batch_data is None or batch_data[0] is None: continue
            inputs, targets, _ = batch_data
            inputs, targets_orig = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            apply_mixup_cutmix = (use_mixup or use_cutmix) and random.random() < mixup_prob
            if apply_mixup_cutmix:
                if use_mixup and use_cutmix:
                    if random.random() < 0.5: inputs, targets_a, targets_b, lam = mixup_data(inputs, targets_orig, mixup_alpha, device)
                    else: inputs, targets_a, targets_b, lam = cutmix_data(inputs, targets_orig, cutmix_alpha, device)
                elif use_mixup: inputs, targets_a, targets_b, lam = mixup_data(inputs, targets_orig, mixup_alpha, device)
                elif use_cutmix: inputs, targets_a, targets_b, lam = cutmix_data(inputs, targets_orig, cutmix_alpha, device)
                else: apply_mixup_cutmix = False
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
                    if is_cosine_scheduler and scheduler is not None: scheduler.step()
                if torch.isnan(loss) or torch.isinf(loss): print(f"\nWARNING: NaN/Inf loss detected E{epoch_num} B{batch_idx+1}. Loss: {loss.item()}. Skipping."); optimizer.zero_grad(set_to_none=True); continue
            batch_loss = loss.item() * gradient_accumulation_steps if is_training and gradient_accumulation_steps > 1 else loss.item()
            running_loss += batch_loss * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1); total_samples += targets_orig.size(0)
            correct_predictions += (predicted == targets_orig).sum().item()
            current_acc = correct_predictions / total_samples if total_samples > 0 else 0
            progress_bar.set_postfix(loss=f"{batch_loss:.4f}", acc=f"{current_acc:.3f}")
    epoch_duration = time.time() - start_time
    if total_samples == 0: print(f"Warning: No valid samples processed in epoch {epoch_num}."); return 0.0, 0.0
    epoch_loss = running_loss / total_samples; epoch_acc = correct_predictions / total_samples
    mode_str = "Training" if is_training else "Validation"
    print(f'\n    Epoch {epoch_num} {mode_str} Summary: Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f} | Duration: {epoch_duration:.2f}s')
    return epoch_loss, epoch_acc


# --- Main Execution Logic (Checkpoint Saving Unchanged) ---
if __name__ == '__main__':
    output_dir = Path('./outputs'); output_dir.mkdir(parents=True, exist_ok=True)
    base_save_path = CONFIG["model"]["base_save_path"]
    base_save_path.mkdir(parents=True, exist_ok=True)
    best_model_path = base_save_path / f"{base_save_path.name}_best.pth"

    if CONFIG["model"]["num_classes"] is None: raise ValueError("Num classes not determined.")

    model = CustomSentinelCNN_v2(
        num_classes=CONFIG["model"]["num_classes"],
        input_channels=CONFIG["model"]["input_channels"] # Should be 16
    )
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG["train"]["label_smoothing"])
    print(f"Base Loss function: CrossEntropyLoss (Label Smoothing={CONFIG['train']['label_smoothing']})")

    optimizer = torch.optim.AdamW( model.parameters(), lr=CONFIG["train"]["lr"], weight_decay=CONFIG["train"]["weight_decay"])
    print(f"Optimizer: AdamW (LR={CONFIG['train']['lr']}, WD={CONFIG['train']['weight_decay']})")

    num_epochs = CONFIG["train"]["epochs"]
    gradient_accumulation_steps = CONFIG["train"]["gradient_accumulation_steps"]
    gradient_clip_norm = CONFIG["train"].get("gradient_clip_norm", None)
    if gradient_clip_norm: print(f"Using gradient clipping with max_norm={gradient_clip_norm}")

    steps_per_epoch = len(train_loader) // gradient_accumulation_steps
    t_max_steps = steps_per_epoch * num_epochs

    if CONFIG["train"]["scheduler"] == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( optimizer, T_max=t_max_steps, eta_min=CONFIG["train"]["eta_min"])
        is_cosine_scheduler = True
        print(f"Using CosineAnnealingLR scheduler (T_max_steps={t_max_steps}, eta_min={CONFIG['train']['eta_min']})")
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=7, verbose=True)
        is_cosine_scheduler = False
        print("Using ReduceLROnPlateau scheduler.")

    patience = CONFIG["train"]["patience"]
    checkpoint_interval = CONFIG["train"]["checkpoint_interval"] # Get interval from config
    print(f"Saving checkpoints every {checkpoint_interval} epochs.")
    overall_best_val_loss = float('inf'); best_model_state_dict = None; total_epochs_run = 0
    scaler = GradScaler(enabled=CONFIG["amp_enabled"]) if DEVICE.type == 'cuda' else None
    if scaler: print("Using Automatic Mixed Precision (AMP).")

    print(f"\n--- Starting Training for {num_epochs} epochs ---")
    epochs_without_improvement = 0
    for epoch in range(num_epochs):
        epoch_num = total_epochs_run + 1
        # Pass full config to run_epoch
        train_loss, train_accuracy = run_epoch( model, train_loader, criterion, optimizer, scaler, scheduler, DEVICE, True,
            epoch_num, num_epochs, CONFIG, is_cosine_scheduler )
        avg_val_loss, val_accuracy = run_epoch( model, val_loader, criterion, None, scaler, None, DEVICE, False,
            epoch_num, num_epochs, CONFIG, False )

        if not is_cosine_scheduler and scheduler is not None: scheduler.step(avg_val_loss)
        print(f'End of Epoch {epoch_num} - Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.4f} | Current LR: {optimizer.param_groups[0]["lr"]:.6e}')

        # --- Checkpointing Logic ---
        is_best = avg_val_loss < overall_best_val_loss
        if is_best:
            overall_best_val_loss = avg_val_loss; epochs_without_improvement = 0
            try: best_model_state_dict = model.state_dict(); torch.save(best_model_state_dict, best_model_path); print(f'---> Validation Loss Improved. Best model saved to {best_model_path}')
            except Exception as e: print(f"Error saving best model: {e}")
        else:
            epochs_without_improvement += 1; print(f'---> Val loss did not improve for {epochs_without_improvement} epochs.')

        if checkpoint_interval > 0 and epoch_num % checkpoint_interval == 0:
            periodic_path = base_save_path / f"{base_save_path.name}_epoch_{epoch_num}.pth"
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
    print("IMPORTANT: Remember to evaluate multiple saved checkpoints (e.g., epoch 5, 10, 15, 20...) on the NPY test set to find the best performing one for that specific dataset.")

