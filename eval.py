import torch
import torch.nn as nn
# Removed optim as it's not needed for evaluation
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models # Added for pre-trained models
import torch.nn.functional as F
from torch.cuda.amp import autocast # Only need autocast for eval

# Removed rasterio as NPY files are loaded directly
import numpy as np
import pandas as pd
import os
# Removed random as it's not typically needed for evaluation logic
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
        "name": "ResNet50_16ch_EuroSAT_v1", # Matches NEW trained model name
        "pretrained": True, # Architecture assumes pretraining logic was used
        "input_channels": 16, # 12 bands + 4 indices
        # !!! Point this to the weights saved by the NEW training script !!!
        "base_save_path": Path('./outputs/resnet50_16ch_eurosat_v1'), # Base path used in NEW training
        # --- Choose which weights to load ---
        "load_weights_name": "ResNet50_16ch_EuroSAT_v1_best.pth", # Load best model based on TIF validation
        # "load_weights_name": "ResNet50_16ch_EuroSAT_v1_epoch_XX.pth", # Example: Load specific epoch checkpoint
        # ----------------------------------
        # !!! IMPORTANT: Set these based on your training output !!!
        "num_classes": 10, # e.g., 10 (Must match the trained model)
        "class_names": ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake'], # (Must match training)
        # Removed dropout params - handled within model definition function
    },
    "data": {
        "prediction_dir": Path('./testset/testset'), # Directory with .npy files for prediction
        "image_size": 224, # Should match image_size used during NEW training (ResNet default)
        "batch_size": 64, # Can often be larger for evaluation (adjust based on VRAM)
        "num_workers": 8, # Adjust based on CPU/system for evaluation
    },
    "device": DEVICE.type,
    "amp_enabled": USE_CUDA, # Enable AMP only on CUDA if desired for eval speed
    "prediction": {
        # Output file path reflects the NEW model
        "predictions_csv_path": Path('./outputs/predictions_resnet50_16ch_eurosat_v1.csv'),
        "kaggle_competition": 'aicrowd-geospatial-challenge', # Replace if needed
        "kaggle_message": 'Submission ResNet50_16ch_EuroSAT_v1', # Update message
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


# --- Data Loading and Preprocessing (Matching NEW training script) ---

# Map for index calculation (relative to the 12 selected bands)
# Copied from modified training script
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
def load_npy_image(filepath: str) -> Optional[np.ndarray]:
    """Loads NPY file, assuming 12 channels (C, H, W) or (H, W, C), returns (12, H, W) NumPy array or None."""
    try:
        filepath_lower = filepath.lower()
        if not filepath_lower.endswith('.npy'): return None
        image = np.load(filepath)

        # Ensure shape is (C, H, W)
        if image.ndim == 3 and image.shape[0] != 12 and image.shape[2] == 12:
            # Input seems to be (H, W, C), transpose it
            image = image.transpose(2, 0, 1)
        elif image.ndim != 3 or image.shape[0] != 12:
             print(f"Error: Unexpected shape/dimensions for NPY {filepath}: {image.shape}. Expected 12 channels (C,H,W) or (H,W,C). Skipping.")
             return None

        # Shape should now be (12, H, W)
        return image.astype(np.float32)
    except Exception as e:
        print(f"Unexpected error loading NPY image {filepath}: {e}")
        traceback.print_exc()
        return None

# --- Index Calculation (operates on UN-NORMALIZED 12 bands) ---
# Copied from modified training script
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
# Copied from modified training script
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

# --- Prediction Dataset Class (NPY Only, Indices BEFORE Norm) ---
class NpyPredictionDataset(Dataset):
    """ Dataset: Loads NPY (12ch), calculates indices (raw), stacks (16ch), normalizes (16ch), transforms. """
    def __init__(self, root_dir: str, transform: Optional[Callable] = None):
        self.root_dir = Path(root_dir); self.transform = transform
        self.output_channels = EVAL_CONFIG["model"]["input_channels"] # Should be 16
        self.file_paths = sorted([p for p in self.root_dir.glob('*.npy')])
        if not self.file_paths: raise FileNotFoundError(f"No .npy files found in {self.root_dir}")
        print(f"Initialized NpyPredictionDataset (Indices BEFORE Norm) with {len(self.file_paths)} files. Output channels: {self.output_channels}")

    def __len__(self): return len(self.file_paths)

    def __getitem__(self, idx: int) -> Optional[Tuple[torch.Tensor, str]]:
        image_path = str(self.file_paths[idx])
        try:
            # 1. Load NPY (12 raw float32 bands)
            image_np_12 = load_npy_image(image_path)
            if image_np_12 is None: return None

            # 2. Calculate Indices from UN-NORMALIZED 12 bands
            indices_dict = calculate_indices_from_raw(image_np_12)
            if not indices_dict: return None

            # 3. Stack 12 bands and 4 indices
            indices_list = [indices_dict['NDVI'], indices_dict['NDWI'], indices_dict['NDBI'], indices_dict['NDRE1']]
            h, w = image_np_12.shape[1], image_np_12.shape[2]
            indices_arrays = [idx_arr[np.newaxis, :h, :w] for idx_arr in indices_list] # Add channel dim

            try:
                image_np_16 = np.concatenate([image_np_12] + indices_arrays, axis=0) # (16, H, W)
            except ValueError as e:
                 print(f"Error concatenating bands and indices for {image_path}: {e}. Shapes: Bands {image_np_12.shape}, Indices {[a.shape for a in indices_arrays]}")
                 return None

            if image_np_16.shape[0] != self.output_channels:
                print(f"Error: Stacked image channels ({image_np_16.shape[0]}) != expected ({self.output_channels}). Skipping.")
                return None

            # 4. Normalize the 16 channels per image
            image_np_norm_16 = normalize_16ch_per_image(image_np_16)
            if image_np_norm_16 is None: return None

            # 5. Convert to tensor
            image_tensor = torch.from_numpy(image_np_norm_16).float()

            # 6. Apply transforms (resize)
            if self.transform:
                image_tensor = self.transform(image_tensor)

            # Check final shape after transform
            if image_tensor.shape[0] != self.output_channels or image_tensor.ndim != 3:
                 print(f"Error: Tensor shape {image_tensor.shape} invalid after transforms for {image_path}. Skipping.")
                 return None


            # Return image tensor and the original path (for ID generation)
            return image_tensor, image_path

        except Exception as e:
            print(f"Error processing prediction image {image_path}: {e}")
            traceback.print_exc()
            return None

# --- Custom Collate Function (Unchanged) ---
def collate_fn(batch):
    """ Filters out None samples and stacks the rest using default_collate. """
    batch = [item for item in batch if item is not None]
    if not batch: return None
    try: return torch.utils.data.dataloader.default_collate(batch)
    except Exception as e: print(f"Error in collate_fn: {e}. Skipping batch."); return None

# --- Transforms (Evaluation only needs resizing) ---
IMG_SIZE = EVAL_CONFIG["data"]["image_size"] # Should be 224
eval_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
])

# --- Model Definition Function (Using Pre-trained ResNet50) ---
# Copied from modified training script
def adapt_resnet_for_multichannel(model_name="resnet50", pretrained=True, num_classes=10, input_channels=16):
    """Loads a pretrained ResNet, adapts first conv layer for N channels, replaces final FC layer."""
    print(f"Loading {'pretrained' if pretrained else 'random weights'} {model_name} for evaluation...")
    # Use weights=None here, we will load state_dict later
    model = models.get_model(model_name, weights=None)

    # --- Adapt the first convolutional layer ---
    original_conv1 = model.conv1
    new_conv1 = nn.Conv2d(input_channels, original_conv1.out_channels,
                          kernel_size=original_conv1.kernel_size,
                          stride=original_conv1.stride,
                          padding=original_conv1.padding,
                          bias=original_conv1.bias is not None)
    # Replace the original conv1 layer - weights will be loaded from state_dict
    model.conv1 = new_conv1
    print(f"Adapted model.conv1 to accept {input_channels} input channels.")

    # --- Replace the final fully connected layer ---
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    print(f"Replaced model.fc layer for {num_classes} output classes.")

    return model


# --- Main Evaluation Function ---
def evaluate_model(config: Dict[str, Any], device: torch.device, num_classes: int, class_names: List[str]):
    """Loads adapted ResNet model, runs prediction on NPY files (indices BEFORE norm), saves CSV."""
    # --- Data ---
    try:
        pred_dataset = NpyPredictionDataset(config["data"]["prediction_dir"], transform=eval_transforms)
        pred_loader = DataLoader( pred_dataset, batch_size=config["data"]["batch_size"], shuffle=False,
            num_workers=config["data"]["num_workers"], pin_memory=True, collate_fn=collate_fn )
    except Exception as e: print(f"Error creating prediction dataset/loader: {e}"); traceback.print_exc(); return

    # --- Model ---
    # Create the correct architecture FIRST
    model = adapt_resnet_for_multichannel(
        model_name="resnet50", # Ensure this matches model used in training
        pretrained=config["model"]["pretrained"], # This influences initial structure slightly if library changes
        num_classes=num_classes,
        input_channels=config["model"]["input_channels"] # Should be 16 channels
    )

    # Load the saved weights
    model_path = config["model"]["load_path"] # Constructed path
    if not model_path.exists(): print(f"ERROR: Model weights file not found at {model_path}"); return
    try:
        print(f"Loading model weights from: {model_path}")
        # Load the state dictionary
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
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
                # Use AMP if enabled
                with autocast(dtype=torch.float16, enabled=AMP_ENABLED):
                    outputs = model(inputs)
                _, predicted_indices = torch.max(outputs, 1)
                predictions.extend(predicted_indices.cpu().numpy())
                # Extract ID like '0001' from path like '.../test_0001.npy'
                batch_ids = [Path(p).stem.replace('test_', '') for p in paths]
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
        # Create DataFrame with 'id', 'label' columns for Kaggle
        pred_df = pd.DataFrame({'id': image_ids, 'label': predicted_class_names})
        # Ensure 'id' column is treated as integer if needed by Kaggle (usually okay as string/object)
        # pred_df['id'] = pd.to_numeric(pred_df['id']) # Optional: convert id to numeric
        csv_path = config["prediction"]["predictions_csv_path"]
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(csv_path, index=False)
        print(f"Predictions saved to: {csv_path}")
    except IndexError as e: print(f"\nError mapping prediction index: {e}. Check num_classes/class_names."); return
    except Exception as e: print(f"\nError creating/saving CSV: {e}"); traceback.print_exc(); return

    # --- Kaggle Submission (Optional - Logic Unchanged) ---
    if config["prediction"]["submit_to_kaggle"]:
        print("\n--- Attempting Kaggle Submission ---")
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
        # Ensure output directory exists
        EVAL_CONFIG["prediction"]["predictions_csv_path"].parent.mkdir(parents=True, exist_ok=True)
        evaluate_model(EVAL_CONFIG, DEVICE, NUM_CLASSES, CLASS_NAMES)
    except Exception as main_e:
        print(f"\nA critical error occurred during script execution: {main_e}")
        traceback.print_exc()
    print("\nEvaluation script finished.")