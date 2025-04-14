import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
from torch.cuda.amp import autocast # Only need autocast for eval

import rasterio # Keep for potential future use
import numpy as np
import pandas as pd
import os
import random
import time
import traceback
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, List, Dict, Any, Optional, Callable
import warnings

# --- Device Setup ---
USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available(): # Check for Apple Silicon GPU
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

# --- Suppress warnings ---
warnings.filterwarnings("ignore")

# --- Configuration ---
EVAL_CONFIG = {
    "model": {
        "name": "CustomSentinelCNN_IdxAfterNorm_v1", # Matches trained model name
        "input_channels": 16, # 12 bands + 4 indices
        # !!! Make sure this points to the weights saved by the corresponding training script !!!
        "base_save_path": Path('./outputs/cnn_idx_after_norm_v1'), # Base path used in training
        # --- Choose which weights to load ---
        "load_weights_name": "cnn_idx_after_norm_v1_best.pth", # Load best model based on TIF validation
        # "load_weights_name": "cnn_idx_after_norm_v1_epoch_40.pth", # Example: Load specific epoch checkpoint
        # ----------------------------------
        # !!! IMPORTANT: Set these based on your training output !!!
        "num_classes": 10, # e.g., 10 (Must match the trained model)
        "class_names": ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake'], # e.g., ['class1', 'class2', ...] (Must match training)
        # Dropout rates used in the trained model's head (copied from training config)
        "dropout_head_1": 0.6,
        "dropout_head_2": 0.4,
    },
    "data": {
        "prediction_dir": Path('./testset/testset'), # Directory with .npy files for prediction
        "image_size": 128, # Should match image_size used during training
        "batch_size": 64, # Can often be larger for evaluation (adjust based on VRAM)
        "num_workers": 8, # Adjust based on CPU/system for evaluation
    },
    "device": DEVICE.type,
    "amp_enabled": USE_CUDA, # Enable AMP only on CUDA if desired for eval speed
    "prediction": {
        "predictions_csv_path": Path('./outputs/predictions_cnn_idx_after_norm_v1.csv'), # Output file
        "kaggle_competition": 'aicrowd-geospatial-challenge', # Replace if needed
        "kaggle_message": 'Submission CustomSentinelCNN_IdxAfterNorm_v1',
        "submit_to_kaggle": False, # Set to False to skip submission attempt
    }
}
# Construct full load path
EVAL_CONFIG["model"]["load_path"] = EVAL_CONFIG["model"]["base_save_path"] / EVAL_CONFIG["model"]["load_weights_name"]


# --- Basic Setup ---
AMP_ENABLED = EVAL_CONFIG["amp_enabled"] and DEVICE.type == 'cuda'
print(f"Automatic Mixed Precision (AMP) enabled for evaluation: {AMP_ENABLED}")

# --- Validate Config ---
if EVAL_CONFIG["model"]["num_classes"] is None or EVAL_CONFIG["model"]["class_names"] is None:
    raise ValueError("Please set 'num_classes' and 'class_names' in EVAL_CONFIG['model'] based on your training run.")
NUM_CLASSES = EVAL_CONFIG["model"]["num_classes"]
CLASS_NAMES = EVAL_CONFIG["model"]["class_names"]
if len(CLASS_NAMES) != NUM_CLASSES:
     raise ValueError(f"Mismatch: num_classes ({NUM_CLASSES}) != length of class_names ({len(CLASS_NAMES)})")
print(f"Expecting {NUM_CLASSES} classes: {CLASS_NAMES}")


# --- Data Loading and Preprocessing ---
# --- Band Selection: Use same indices as training script ---
TARGET_BANDS_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12] # Excludes index 9 (10th band)
# Map for index calculation (relative to the 12 selected bands)
BAND_MAP = {
    "B1_Coastal": 0, "B2_Blue": 1, "B3_Green": 2, "B4_Red": 3, "B5_RE1": 4, # Red Edge 1
    "B6_RE2": 5, "B7_RE3": 6, "B8_NIR": 7, "B9_WV": 8, "B11_SWIR1": 9,
    "B12_SWIR2": 10, "B8A_NIR2": 11
}

def load_sentinel2_image(filepath: str) -> Optional[np.ndarray]:
    """Loads NPY file, ensures 12 specific bands, returns (12, H, W) NumPy array or None."""
    global TARGET_BANDS_INDICES
    try:
        filepath_lower = filepath.lower(); image_data = None
        if not filepath_lower.endswith('.npy'): return None
        image = np.load(filepath); processed_image = None
        if image.ndim == 3 and image.shape[0] in [12, 13]: processed_image = image
        elif image.ndim == 3 and image.shape[2] in [12, 13]: processed_image = image.transpose(2, 0, 1)
        else: print(f"Error: Unexpected shape/dimensions for NPY {filepath}: {image.shape}. Skipping."); return None
        if processed_image.shape[0] == 13: image_data = processed_image[TARGET_BANDS_INDICES, :, :]
        elif processed_image.shape[0] == 12: image_data = processed_image
        else: print(f"Error: Processed image has unexpected channel count ({processed_image.shape[0]}) for {filepath}. Skipping."); return None
        if image_data is None or image_data.shape[0] != 12: return None
        return image_data.astype(np.float32)
    except Exception as e: print(f"Unexpected error loading NPY image {filepath}: {e}"); traceback.print_exc(); return None

# --- Normalization for 12 channels (Matches training script) ---
def normalize_image_per_image(image_np: np.ndarray) -> Optional[np.ndarray]:
    """Normalizes 12-channel NumPy image (C, H, W) using its own stats."""
    if image_np is None or image_np.ndim != 3 or image_np.shape[0] != 12: return None
    mean = np.nanmean(image_np, axis=(1, 2), keepdims=True); std = np.nanstd(image_np, axis=(1, 2), keepdims=True)
    mean = np.nan_to_num(mean, nan=0.0); std = np.nan_to_num(std, nan=1.0)
    std_safe = std + 1e-7; normalized_image = (image_np - mean) / std_safe
    normalized_image = np.clip(normalized_image, -6.0, 6.0)
    if np.isnan(normalized_image).any() or np.isinf(normalized_image).any():
        normalized_image = np.nan_to_num(normalized_image, nan=0.0, posinf=0.0, neginf=0.0)
    return normalized_image

# --- Index Calculation (operates on NORMALIZED 12 bands - Matches training script) ---
def calculate_indices(image_np_12bands_normalized: np.ndarray, epsilon=1e-6) -> Dict[str, np.ndarray]:
    """Calculates NDVI, NDWI, NDBI, NDRE1 from a NORMALIZED 12-band NumPy array. Clips output."""
    indices = {}; clip_val = 10.0
    try:
        nir = image_np_12bands_normalized[BAND_MAP["B8_NIR"], :, :]; red = image_np_12bands_normalized[BAND_MAP["B4_Red"], :, :]
        green = image_np_12bands_normalized[BAND_MAP["B3_Green"], :, :]; swir1 = image_np_12bands_normalized[BAND_MAP["B11_SWIR1"], :, :]
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

# --- Prediction Dataset Class (NPY Only, Indices AFTER Norm) ---
class NpyPredictionDataset(Dataset):
    """ Dataset: Loads NPY, normalizes 12 bands, calculates indices, stacks to 16ch. """
    def __init__(self, root_dir: str, transform: Optional[Callable] = None):
        self.root_dir = Path(root_dir); self.transform = transform
        self.output_channels = EVAL_CONFIG["model"]["input_channels"] # Should be 16
        self.file_paths = sorted([p for p in self.root_dir.glob('*.npy')])
        if not self.file_paths: raise FileNotFoundError(f"No .npy files found in {self.root_dir}")
        print(f"Initialized NpyPredictionDataset (Indices After Norm) with {len(self.file_paths)} files. Output channels: {self.output_channels}")
    def __len__(self): return len(self.file_paths)
    def __getitem__(self, idx: int) -> Optional[Tuple[torch.Tensor, str]]:
        image_path = str(self.file_paths[idx])
        try:
            # 1. Load NPY (12 raw-ish bands)
            image_np_12 = load_sentinel2_image(image_path)
            if image_np_12 is None: return None
            # 2. Normalize the 12 bands
            image_np_norm_12 = normalize_image_per_image(image_np_12)
            if image_np_norm_12 is None: return None
            # 3. Calculate Indices from NORMALIZED bands
            indices_dict = calculate_indices(image_np_norm_12)
            if not indices_dict: return None
            # 4. Stack normalized bands and indices
            indices_list = [indices_dict['NDVI'], indices_dict['NDWI'], indices_dict['NDBI'], indices_dict['NDRE1']]
            indices_arrays = [idx[np.newaxis, :, :] for idx in indices_list]
            final_image_np = np.concatenate([image_np_norm_12] + indices_arrays, axis=0) # (16, H, W)
            if final_image_np.shape[0] != self.output_channels: return None
            # 5. Convert to tensor
            image_tensor = torch.from_numpy(final_image_np).float()
            # 6. Apply transforms (resize)
            if self.transform: image_tensor = self.transform(image_tensor)
            return image_tensor, image_path
        except Exception as e: print(f"Error processing prediction image {image_path}:"); traceback.print_exc(); return None

# --- Custom Collate Function ---
def collate_fn(batch):
    """ Filters out None samples and stacks the rest using default_collate. """
    batch = [item for item in batch if item is not None]
    if not batch: return None
    try: return torch.utils.data.dataloader.default_collate(batch)
    except Exception as e: print(f"Error in collate_fn: {e}. Skipping batch."); return None

# --- Transforms (Evaluation only needs resizing) ---
IMG_SIZE = EVAL_CONFIG["data"]["image_size"]
eval_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
])

# --- Custom Model Definition (ResNet-like Blocks - Copied from training script) ---
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
class CustomSentinelCNN_v2(nn.Module): # Name matches training script
    """Custom CNN with ResNet-like blocks (must match training script)."""
    def __init__(self, num_classes: int, input_channels: int = EVAL_CONFIG["model"]["input_channels"]): # Use config input channels
        super().__init__()
        print(f"Creating CustomSentinelCNN_v2 for evaluation: {input_channels} channels, {num_classes} classes.")
        if input_channels <= 0: raise ValueError("input_channels must be positive.")
        self.stem = nn.Sequential( nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(64), Mish(), nn.MaxPool2d(kernel_size=2, stride=2) )
        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, num_blocks=2, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1); self.max_pool = nn.AdaptiveMaxPool2d(1); pooled_features = 512
        # Use dropout rates defined in config (matching training)
        dropout1 = EVAL_CONFIG["model"].get("dropout_head_1", 0.6)
        dropout2 = EVAL_CONFIG["model"].get("dropout_head_2", 0.4)
        self.head = nn.Sequential( nn.Flatten(), nn.BatchNorm1d(pooled_features * 2), nn.Dropout(dropout1),
            nn.Linear(pooled_features * 2, 256), Mish(), nn.BatchNorm1d(256), nn.Dropout(dropout2),
            nn.Linear(256, num_classes) )
        # No weight initialization needed
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []; layers.append(ResidualConvBlock(in_channels, out_channels, stride=stride))
        for _ in range(1, num_blocks): layers.append(ResidualConvBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.stem(x); x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        avg_p = self.avg_pool(x); max_p = self.max_pool(x); x = torch.cat((avg_p, max_p), dim=1); x = self.head(x)
        return x

# --- Main Evaluation Function ---
def evaluate_model(config: Dict[str, Any], device: torch.device, num_classes: int, class_names: List[str]):
    """Loads model, runs prediction on NPY files (indices AFTER norm), saves CSV."""
    # --- Data ---
    try:
        pred_dataset = NpyPredictionDataset(config["data"]["prediction_dir"], transform=eval_transforms)
        pred_loader = DataLoader( pred_dataset, batch_size=config["data"]["batch_size"], shuffle=False,
            num_workers=config["data"]["num_workers"], pin_memory=True, collate_fn=collate_fn )
    except Exception as e: print(f"Error creating prediction dataset/loader: {e}"); traceback.print_exc(); return

    # --- Model ---
    model = CustomSentinelCNN_v2(num_classes=num_classes, input_channels=config["model"]["input_channels"]) # Should be 16 channels
    model_path = config["model"]["load_path"] # Constructed path
    if not model_path.exists(): print(f"ERROR: Model weights file not found at {model_path}"); return
    try:
        print(f"Loading model weights from: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device); model.eval()
        print("Model loaded successfully.")
    except Exception as e: print(f"Error loading model state_dict: {e}"); traceback.print_exc(); return

    # --- Prediction ---
    predictions = []; image_ids = []
    print("\n--- Starting Prediction ---")
    with torch.no_grad():
        progress_bar = tqdm(pred_loader, desc="Predicting")
        for batch_data in progress_bar:
            if batch_data is None: print("Warning: Skipping None batch."); continue
            try:
                inputs, paths = batch_data # Unpack image tensors (16 channels) and paths
                inputs = inputs.to(device, non_blocking=True)
                with autocast(dtype=torch.float16, enabled=AMP_ENABLED):
                    outputs = model(inputs)
                _, predicted_indices = torch.max(outputs, 1)
                predictions.extend(predicted_indices.cpu().numpy())
                batch_ids = [Path(p).stem for p in paths] # Get stem like 'test_0001'
                image_ids.extend(batch_ids)
            except Exception as e: print(f"\nError during prediction batch: {e}"); traceback.print_exc()

    if not predictions or not image_ids: print("\nError: No predictions or image IDs generated."); return

    # --- Generate Submission File ---
    print(f"\nGenerated {len(predictions)} predictions for {len(image_ids)} images.")
    if len(predictions) != len(image_ids):
         print(f"Warning: Mismatch predictions vs image IDs ({len(predictions)} vs {len(image_ids)})."); min_len = min(len(predictions), len(image_ids))
         predictions = predictions[:min_len]; image_ids = image_ids[:min_len]; print(f"Proceeding with {min_len} pairs.")
    try:
        predicted_class_names = [class_names[idx] for idx in predictions]
        pred_df = pd.DataFrame({'test_id': image_ids, 'label': predicted_class_names})
        # Remove "test_" prefix
        pred_df['test_id'] = pred_df['test_id'].str.replace('test_', '', regex=False)
        print("Removed 'test_' prefix from test_id column.")
        csv_path = config["prediction"]["predictions_csv_path"]
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(csv_path, index=False)
        print(f"Predictions saved to: {csv_path}")
    except IndexError as e: print(f"\nError mapping prediction index: {e}. Check num_classes/class_names."); return
    except Exception as e: print(f"\nError creating/saving CSV: {e}"); traceback.print_exc(); return

    # --- Kaggle Submission (Optional) ---
    if config["prediction"]["submit_to_kaggle"]:
        print("\n--- Attempting Kaggle Submission ---")
        # (Kaggle submission logic remains the same)
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi(); api.authenticate()
            competition = config["prediction"]["kaggle_competition"]
            message = config["prediction"]["kaggle_message"]
            if not competition: print("Kaggle competition slug not set. Skipping."); return
            print(f"Submitting {csv_path} to competition: {competition}")
            api.competition_submit(file_name=str(csv_path), message=message, competition=competition)
            print("Submission successful!")
        except ImportError: print("Kaggle API not found. Install with: pip install kaggle")
        except Exception as e: print(f"Kaggle API submission failed: {e}")

# --- Run Evaluation ---
if __name__ == '__main__':
    try:
        EVAL_CONFIG["prediction"]["predictions_csv_path"].parent.mkdir(parents=True, exist_ok=True)
        evaluate_model(EVAL_CONFIG, DEVICE, NUM_CLASSES, CLASS_NAMES)
    except Exception as main_e:
        print(f"\nA critical error occurred during script execution: {main_e}")
        traceback.print_exc()
    print("\nEvaluation script finished.")

