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
elif torch.backends.mps.is_available():
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
TARGET_BANDS_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12]

# --- Band Mapping (Standard Order) ---
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
            all_bands = src.read(list(range(1, 14))).astype(np.float32)
        standard_order_indices_in_tif = [0, 1, 2, 3, 4, 5, 6, 7, 12, 8, 10, 11] # B1..B8, B8A, B9, B11, B12
        image_data_12_standard_order = all_bands[standard_order_indices_in_tif, :, :]
        if image_data_12_standard_order.shape[0] != 12: return None
        return image_data_12_standard_order
    except Exception as e: print(f"Error loading/reordering TIF {filepath}: {e}"); return None

# --- Index Calculation ---
def calculate_indices_from_raw(image_np_12bands: np.ndarray, epsilon=1e-7) -> Dict[str, np.ndarray]:
    indices = {}; clip_val = 1.0; global BAND_MAP_12
    try:
        # (Index calculation logic remains the same - uses BAND_MAP_12)
        nir=image_np_12bands[BAND_MAP_12["B8_NIR"],:,:]; red=image_np_12bands[BAND_MAP_12["B4_Red"],:,:]
        green=image_np_12bands[BAND_MAP_12["B3_Green"],:,:]; swir1=image_np_12bands[BAND_MAP_12["B11_SWIR1"],:,:]
        re1=image_np_12bands[BAND_MAP_12["B5_RE1"],:,:]
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
        mean=np.nan_to_num(mean, nan=0.0); std=np.nan_to_num(std, nan=1.0); std[std < 1e-7]=1.0
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
        self.output_channels = output_channels
        print(f"Initialized Sentinel2Dataset with {len(self.paths_labels)} samples. Output channels: {self.output_channels}")
    def __len__(self): return len(self.paths_labels)
    def __getitem__(self, idx):
        image_path, label = self.paths_labels[idx]
        try:
            image_np_12 = load_sentinel2_image(image_path);
            if image_np_12 is None: return None
            indices_dict = calculate_indices_from_raw(image_np_12);
            if not indices_dict: return None
            indices_list=[indices_dict['NDVI'], indices_dict['NDWI'], indices_dict['NDBI'], indices_dict['NDRE1']]
            h,w=image_np_12.shape[1], image_np_12.shape[2]; indices_arrays=[idx_arr[np.newaxis, :h, :w] for idx_arr in indices_list]
            try: image_np_16 = np.concatenate([image_np_12] + indices_arrays, axis=0)
            except ValueError as e: print(f"Concat Error {image_path}: {e}"); return None
            if image_np_16.shape[0] != self.output_channels: return None
            image_np_norm_16 = normalize_16ch_per_image(image_np_16);
            if image_np_norm_16 is None: return None
            image_tensor = torch.from_numpy(image_np_norm_16).float()
            if self.transform: image_tensor = self.transform(image_tensor)
            if image_tensor.shape[0]!=self.output_channels or image_tensor.ndim!=3: return None
            label_tensor = torch.tensor(label, dtype=torch.long)
            return image_tensor, label_tensor, image_path
        except Exception as e: print(f"Error processing image {image_path}: {e}"); return None

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
class_names = []
idx_counter = 0
train_root_dir = Path(config["data"]["train_dir"])
if not train_root_dir.exists(): raise FileNotFoundError(f"Training directory not found: {train_root_dir}")
for class_folder in sorted(train_root_dir.iterdir()):
     if class_folder.is_dir() and not class_folder.name.startswith('.'):
         class_name = class_folder.name
         if class_name not in class_to_idx_map: class_to_idx_map[class_name] = idx_counter; class_names.append(class_name); idx_counter += 1
         class_idx = class_to_idx_map[class_name]
         for filepath in class_folder.glob('*.tif'): full_dataset_paths_labels.append((str(filepath), class_idx))

num_classes = len(class_names)
if num_classes == 0: raise FileNotFoundError(f"No valid class folders found in {train_root_dir}")
config["model"]["num_classes"] = num_classes
config["model"]["class_names"] = class_names
print(f"Found {len(full_dataset_paths_labels)} training paths in {num_classes} classes: {class_names}")
print(f"Class mapping: {class_to_idx_map}")

try:
    train_info, val_info = train_test_split(
        full_dataset_paths_labels,
        train_size=float(config["data"]["train_ratio"]), # Cast train_ratio
        random_state=int(config["train"]["seed"]),       # Cast seed
        stratify=[label for _, label in full_dataset_paths_labels]
    )
except ValueError as e:
     print(f"Warning: Stratified split failed ({e}). Falling back to non-stratified split.")
     train_info, val_info = train_test_split(
         full_dataset_paths_labels,
         train_size=float(config["data"]["train_ratio"]), # Cast train_ratio
         random_state=int(config["train"]["seed"])       # Cast seed
     )

print("Creating Sentinel2Dataset instances...")
train_dataset = Sentinel2Dataset(
    train_info,
    transform=train_transforms,
    output_channels=int(config["model"]["input_channels"]) # Cast channels
)
val_tif_dataset = Sentinel2Dataset(
    val_info,
    transform=val_transforms,
    output_channels=int(config["model"]["input_channels"]) # Cast channels
)

# --- Create DataLoaders ---
def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch: return None, None, None
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
    pin_memory=True,
    collate_fn=collate_fn,
    drop_last=True,
    persistent_workers=persistent_workers
)
val_loader = DataLoader(
    val_tif_dataset,
    batch_size=int(config["data"]["batch_size"])*2, # Cast batch_size
    shuffle=False,
    num_workers=int(config["data"]["num_workers"]), # Cast num_workers
    pin_memory=True,
    collate_fn=collate_fn,
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
    model = models.get_model(model_name, weights=weights)
    original_conv1 = model.conv1
    new_conv1 = nn.Conv2d(input_channels, original_conv1.out_channels, kernel_size=original_conv1.kernel_size, stride=original_conv1.stride, padding=original_conv1.padding, bias=original_conv1.bias is not None)
    if pretrained:
        print(f"Adapting pretrained weights for conv1 from 3 to {input_channels} channels...")
        original_weights = original_conv1.weight.data
        avg_weights = torch.mean(original_weights, dim=1, keepdim=True)
        repeated_weights = avg_weights.repeat(1, input_channels, 1, 1)
        new_conv1.weight.data = repeated_weights
        if new_conv1.bias is not None: new_conv1.bias.data = original_conv1.bias.data
        print("Pretrained conv1 weights adapted.")
    else:
        print("Initializing conv1 weights randomly...")
        nn.init.kaiming_normal_(new_conv1.weight, mode='fan_out', nonlinearity='relu')
        if new_conv1.bias is not None: nn.init.zeros_(new_conv1.bias)
        print("Random conv1 weights initialized.")
    model.conv1 = new_conv1
    print("Replacing and initializing final fc layer...")
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    nn.init.xavier_uniform_(model.fc.weight);
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
def rand_bbox(size, lam):
    W = size[2]; H = size[3]; cut_rat = np.sqrt(1. - lam); cut_w = int(W * cut_rat); cut_h = int(H * cut_rat)
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
    if optimizer is None: return 0.0
    for param_group in optimizer.param_groups: return param_group['lr']
def adjust_lr_with_warmup(optimizer, initial_lr, epoch, step, steps_per_epoch, warmup_epochs, config):
    total_warmup_steps = warmup_epochs * steps_per_epoch
    current_step = epoch * steps_per_epoch + step
    if current_step < total_warmup_steps:
        start_lr = 1e-7
        lr = start_lr + (initial_lr - start_lr) * (current_step + 1) / total_warmup_steps
        for param_group in optimizer.param_groups: param_group['lr'] = lr
        return lr
    else: return None

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
    amp_enabled = scaler is not None and bool(config['amp_enabled']) # Cast amp_enabled
    use_mixup = bool(config['train']['use_mixup']) and is_training
    use_cutmix = bool(config['train']['use_cutmix']) and is_training
    mixup_alpha = float(config['train']['mixup_alpha'])
    cutmix_alpha = float(config['train']['cutmix_alpha'])
    mixup_prob = float(config['train']['mixup_prob'])
    label_smoothing = float(config['train']['label_smoothing'])
    # --- End Type Casting ---

    if is_training: model.train(); print(f'---> Training Epoch {epoch_num}/{total_epochs} | Target LR: {initial_lr:.4e}')
    else: model.eval(); print(f'---> Validation Epoch {epoch_num}/{total_epochs}')

    running_loss = 0.0; correct_predictions = 0; total_samples = 0; start_time = time.time()
    base_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing, reduction='mean')
    progress_bar = tqdm(loader, desc=f"Epoch {epoch_num} {'Train' if is_training else 'Val'}", leave=False)

    with torch.set_grad_enabled(is_training):
        for batch_idx, batch_data in enumerate(progress_bar):
            if batch_data is None or batch_data[0] is None: continue
            inputs, targets, _ = batch_data
            inputs, targets_orig = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            current_lr = None
            # Pass integer warmup_epochs here
            if is_training and warmup_epochs > 0 and (epoch_num-1) < warmup_epochs :
                 current_lr = adjust_lr_with_warmup(optimizer, initial_lr, epoch_num-1, batch_idx, steps_per_epoch, warmup_epochs, config)

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
                # Use integer gradient_accumulation_steps
                if is_training and gradient_accumulation_steps > 1: loss = loss / gradient_accumulation_steps

            if is_training:
                if amp_enabled: scaler.scale(loss).backward()
                else: loss.backward()
                # Use integer gradient_accumulation_steps
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    if amp_enabled: scaler.unscale_(optimizer);
                    # Use float gradient_clip_norm if not None
                    if gradient_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
                    if amp_enabled: scaler.step(optimizer); scaler.update()
                    else: optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    # Use integer warmup_epochs
                    is_after_warmup = (warmup_epochs == 0) or ((epoch_num-1) >= warmup_epochs)
                    if is_after_warmup and is_cosine_scheduler and scheduler is not None:
                         scheduler.step()

                if torch.isnan(loss) or torch.isinf(loss): print(f"\nWARNING: NaN/Inf loss {loss.item()} E{epoch_num} B{batch_idx+1}. Skipping."); optimizer.zero_grad(set_to_none=True); continue

            # Use integer gradient_accumulation_steps
            batch_loss = loss.item() * gradient_accumulation_steps if is_training and gradient_accumulation_steps > 1 else loss.item()
            running_loss += batch_loss * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1); total_samples += targets_orig.size(0)
            correct_predictions += (predicted == targets_orig).sum().item()
            current_acc = correct_predictions / total_samples if total_samples > 0 else 0
            if is_training: display_lr = get_lr(optimizer); progress_bar.set_postfix(loss=f"{batch_loss:.4f}", acc=f"{current_acc:.3f}", lr=f"{display_lr:.2e}")
            else: progress_bar.set_postfix(loss=f"{batch_loss:.4f}", acc=f"{current_acc:.3f}")

    epoch_duration = time.time() - start_time
    if total_samples == 0: print(f"Warning: No valid samples processed in epoch {epoch_num}."); return 0.0, 0.0
    epoch_loss = running_loss / total_samples; epoch_acc = correct_predictions / total_samples
    mode_str = "Training" if is_training else "Validation"
    final_lr = None
    if is_training: final_lr = get_lr(optimizer); print(f'\n    Epoch {epoch_num} {mode_str} Summary: Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f} | Final LR: {final_lr:.4e} | Duration: {epoch_duration:.2f}s')
    else: print(f'\n    Epoch {epoch_num} {mode_str} Summary: Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f} | Duration: {epoch_duration:.2f}s')
    return epoch_loss, epoch_acc

# --- Main Execution Logic ---
if __name__ == '__main__':
    # Paths from config
    base_save_path = Path(config["model"]["base_save_path"])
    base_save_path.mkdir(parents=True, exist_ok=True)
    best_model_path = base_save_path / f"{config['model']['name']}_best_non_swa.pth"
    swa_model_path = base_save_path / f"{config['model']['name']}_swa_final.pth"

    if config["model"]["num_classes"] is None: raise ValueError("Num classes not determined.")
    num_classes = int(config["model"]["num_classes"]) # Cast num_classes

    # Instantiate model using config
    model = adapt_resnet_for_multichannel(
        model_name=config["model"].get("architecture", "resnet50"),
        pretrained=bool(config["model"]["pretrained"]), # Cast pretrained
        num_classes=num_classes,
        input_channels=int(config["model"]["input_channels"]) # Cast channels
    )
    model.to(DEVICE)

    # SWA Model Initialization
    swa_model = AveragedModel(model)
    print("SWA Model wrapper created.")

    # Optimizer and Loss from config
    criterion = nn.CrossEntropyLoss(label_smoothing=float(config["train"]["label_smoothing"])) # Cast smoothing
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["lr"]),             # Cast lr
        weight_decay=float(config["train"]["weight_decay"]) # Cast wd
    )
    print(f"Optimizer: AdamW (Target Initial LR={config['train']['lr']}, WD={config['train']['weight_decay']})")

    # Cast warmup_epochs
    warmup_epochs = int(config["train"]["warmup_epochs"])
    if warmup_epochs > 0:
        print(f"Using {warmup_epochs} warm-up epochs.")
        for param_group in optimizer.param_groups: param_group['lr'] = 1e-7 # Start low

    # Scheduler Setup from config
    num_epochs = int(config["train"]["epochs"]) # Cast epochs
    gradient_accumulation_steps = int(config["train"]["gradient_accumulation_steps"]) # Cast steps
    gradient_clip_norm_config = config["train"].get("gradient_clip_norm") # Handle None case
    gradient_clip_norm = float(gradient_clip_norm_config) if gradient_clip_norm_config is not None else None
    if gradient_clip_norm: print(f"Using gradient clipping with max_norm={gradient_clip_norm}")

    steps_per_epoch = len(train_loader) // gradient_accumulation_steps
    if steps_per_epoch == 0: steps_per_epoch = 1

    # Main LR Scheduler
    swa_start_epoch = int(config["train"]["swa_start_epoch"]) # Cast swa start epoch
    epochs_before_swa = swa_start_epoch - 1
    steps_before_swa = steps_per_epoch * (epochs_before_swa - warmup_epochs)
    if steps_before_swa <= 0: steps_before_swa = 1

    scheduler = None
    is_cosine_scheduler = False
    if config["train"]["scheduler"] == "CosineAnnealingLR":
        eta_min_config = config["train"]["eta_min"]
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=steps_before_swa,
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
        print(f"Warning: Unknown scheduler type '{config['train']['scheduler']}'. No main scheduler used.")


    # SWA Scheduler
    swa_lr_config = config["train"]["swa_lr"]
    anneal_epochs_config = config["train"]["swa_anneal_epochs"]
    swa_scheduler = SWALR(
        optimizer,
        swa_lr=float(swa_lr_config),                 # Cast swa_lr
        anneal_epochs=int(anneal_epochs_config),     # Cast anneal_epochs
        anneal_strategy='cos'
    )
    print(f"SWA will start at epoch {swa_start_epoch} with SWA LR {float(swa_lr_config):.2e}")

    # Training Loop Setup
    patience = int(config["train"]["patience"]) # Cast patience
    checkpoint_interval = int(config["train"]["checkpoint_interval"]) # Cast interval
    overall_best_val_loss = float('inf'); total_epochs_run = 0
    scaler = GradScaler(enabled=bool(config["amp_enabled"])) # Cast amp_enabled
    if scaler.is_enabled(): print("Using Automatic Mixed Precision (AMP).")

    print(f"\n--- Starting Training for {num_epochs} epochs ---")
    epochs_without_improvement = 0

    # --- Training Loop ---
    for epoch in range(num_epochs):
        epoch_num = total_epochs_run + 1
        use_swa_scheduler = epoch_num >= swa_start_epoch
        current_scheduler = swa_scheduler if use_swa_scheduler else scheduler
        current_is_cosine = (not use_swa_scheduler) and is_cosine_scheduler

        if use_swa_scheduler and epoch_num == swa_start_epoch:
            print(f"--- Epoch {epoch_num}: Switching to SWA Scheduler ---")

        # Pass the full config dict
        train_loss, train_accuracy = run_epoch( model, train_loader, criterion, optimizer, scaler, current_scheduler, DEVICE, True, epoch_num, num_epochs, config, steps_per_epoch, current_is_cosine )
        avg_val_loss, val_accuracy = run_epoch( model, val_loader, criterion, None, scaler, None, DEVICE, False, epoch_num, num_epochs, config, steps_per_epoch, False )

        # Step epoch-based schedulers (AFTER warm-up)
        is_after_warmup = (warmup_epochs == 0) or (epoch_num > warmup_epochs)
        if is_after_warmup:
            if use_swa_scheduler:
                 swa_scheduler.step() # SWA scheduler steps per epoch
            elif not current_is_cosine and scheduler is not None: # Step plateau scheduler
                 scheduler.step(avg_val_loss)

        # SWA Model Averaging (Implicit)
        if epoch_num >= swa_start_epoch:
             if epoch_num == swa_start_epoch: print(f"--- Epoch {epoch_num}: SWA phase started. Averaging weights... ---")

        # Checkpointing based on non-SWA model validation loss
        is_best = avg_val_loss < overall_best_val_loss
        if is_best:
            overall_best_val_loss = avg_val_loss; epochs_without_improvement = 0
            try: torch.save(model.state_dict(), best_model_path); print(f'---> Best non-SWA model state saved to {best_model_path}')
            except Exception as e: print(f"Error saving best non-SWA model state_dict: {e}")
        else:
            epochs_without_improvement += 1; print(f'---> Val loss did not improve for {epochs_without_improvement} epochs.')

        # Periodic checkpoint saving (non-SWA model)
        if checkpoint_interval > 0 and epoch_num % checkpoint_interval == 0:
             periodic_path = base_save_path / f"{config['model']['name']}_epoch_{epoch_num}.pth"
             try: torch.save(model.state_dict(), periodic_path); print(f'---> Periodic checkpoint saved to {periodic_path}')
             except Exception as e: print(f"Error saving periodic checkpoint: {e}")

        # Early stopping check
        if epochs_without_improvement >= patience: print(f'Early stopping triggered at epoch {epoch_num}.'); break
        total_epochs_run = epoch_num

    # --- After Training Loop ---
    print("\n--- Training Complete ---")
    if total_epochs_run >= swa_start_epoch:
        print("Updating SWA Batch Norm statistics...")
        bn_update_device = DEVICE if DEVICE else None
        try:
            torch.optim.swa_utils.update_bn(train_loader, swa_model, device=bn_update_device)
            print("SWA Batch Norm update complete.")
            print(f"Saving final SWA model state to {swa_model_path}")
            torch.save(swa_model.module.state_dict(), swa_model_path) # Save the underlying averaged model
        except Exception as e: print(f"Error during SWA BN update or saving: {e}")
    else: print(f"SWA phase (start epoch {swa_start_epoch}) not reached.")

    if overall_best_val_loss != float('inf'): print(f'Best non-SWA model state saved to {best_model_path} (Val Loss: {overall_best_val_loss:.4f})')
    print("\n--- Script Finished ---")
    print(f"IMPORTANT: Evaluate the desired checkpoint (e.g., {best_model_path} or {swa_model_path}) using the evaluation script.")

