import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
# Use torch.amp directly for GradScaler and autocast
# Note: GradScaler is typically used for training, not evaluation, but autocast is useful.
from torch.cuda.amp import autocast, GradScaler # Keep GradScaler import for context if needed, but won't be used

import rasterio
import numpy as np
import pandas as pd
import os
import random
import time
import traceback
from pathlib import Path
# Use standard tqdm
from tqdm import tqdm
from typing import Tuple, List, Dict, Any, Optional, Callable

import warnings

# --- Suppress warnings (optional) ---
# warnings.filterwarnings("ignore")

# --- Configuration (Adjust as needed) ---
EVAL_CONFIG = {
    "model": {
        # !!! Point this to the actual saved weights of your trained Sentinel2Classifier !!!
        "load_path": Path('./sentinel2_classifier_best.pth'), # Example path from training script
        # !!! IMPORTANT: Set these based on your training output !!!
        "num_classes": 10, # e.g., 10 (Must match the trained model)
        "class_names": ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake'], # e.g., ['class1', 'class2', ...] (Must match training)
    },
    "data": {
        "prediction_dir": Path('./testset/testset'), # Directory with .npy files
        "num_workers": 4,
        "image_size": 128, # Should match training image size
        "batch_size": 32, # Can often be larger for evaluation
    },
    "device": "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cpu",
    "amp_enabled": True, # Enable AMP if using CUDA and want potential speedup
    "prediction": {
        "predictions_csv_path": Path('./outputs/predictions_sentinel2_classifier.csv'), # Updated output file name
        "kaggle_competition": 'aicrowd-geospatial-challenge', # Replace if needed
        "kaggle_message": 'Submission with Sentinel2Classifier', # Updated message
        "submit_to_kaggle": False, # Set to False to skip submission attempt
    }
}

# --- Basic Setup ---
DEVICE = torch.device(EVAL_CONFIG["device"])
# AMP is enabled only if configured AND on CUDA device
AMP_ENABLED = EVAL_CONFIG["amp_enabled"] and DEVICE.type == 'cuda'
print(f"Using device: {DEVICE}")
print(f"Automatic Mixed Precision (AMP) enabled for evaluation: {AMP_ENABLED}")

# --- Make sure NUM_CLASSES and CLASS_NAMES are set ---
if EVAL_CONFIG["model"]["num_classes"] is None or EVAL_CONFIG["model"]["class_names"] is None:
    raise ValueError("Please set 'num_classes' and 'class_names' in EVAL_CONFIG['model'] based on your training run.")
NUM_CLASSES = EVAL_CONFIG["model"]["num_classes"]
CLASS_NAMES = EVAL_CONFIG["model"]["class_names"]
# Validate length consistency
if len(CLASS_NAMES) != NUM_CLASSES:
     raise ValueError(f"Mismatch: num_classes ({NUM_CLASSES}) != length of class_names ({len(CLASS_NAMES)})")
print(f"Expecting {NUM_CLASSES} classes: {CLASS_NAMES}")


# --- Data Loading Function (Copied from training script) ---
def load_sentinel2_image(filepath: str) -> Optional[np.ndarray]:
    """Loads a Sentinel-2 image (TIF or NPY), returns NumPy CHW (12 bands) or None on error."""
    try:
        filepath_lower = filepath.lower()
        if filepath_lower.endswith('.tif') or filepath_lower.endswith('.tiff'):
            with rasterio.open(filepath) as src:
                bands_to_read = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13] # Assuming 1-based indexing in rasterio
                if src.count < 13:
                    print(f"Warning: Expected >=13 bands, got {src.count} in {filepath}. Reading available bands.")
                    bands_to_read = list(range(1, min(src.count + 1, 13)))
                    if len(bands_to_read) < 12:
                         print(f"Error: Not enough bands ({len(bands_to_read)}) to form 12 channels in {filepath}. Skipping.")
                         return None
                image = src.read(bands_to_read)

        elif filepath_lower.endswith('.npy'):
            image = np.load(filepath)
            if image.shape[0] != 12:
                 if image.shape[0] == 13:
                     # Assume standard Sentinel-2 order and drop B10 (index 9)
                     indices_to_keep = list(range(9)) + [10, 11, 12]
                     image = image[indices_to_keep, :, :]
                 # Check if it's HWC and transpose
                 elif image.ndim == 3 and image.shape[2] == 12:
                     print(f"Info: Transposing NPY from (H, W, C) to (C, H, W) for {filepath}")
                     image = np.transpose(image, (2, 0, 1))
                 else:
                    print(f"Error: Unexpected shape for .npy {filepath}: {image.shape}. Expected (12, H, W) or (13, H, W) or (H, W, 12). Skipping.")
                    return None
        else:
            print(f"Error: Unsupported file type: {filepath}. Skipping.")
            return None
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

# --- Per-Image Normalization Function (Copied from training script) ---
def normalize_image_per_image(image_np: np.ndarray) -> Optional[np.ndarray]:
    """Normalizes 12-channel NumPy image (C, H, W) using its own stats."""
    if image_np is None: return None
    if image_np.ndim != 3 or image_np.shape[0] != 12:
        print(f"Error: Invalid shape for per-image normalization: {image_np.shape}. Expected (12, H, W). Skipping normalization.")
        return None

    mean = np.nanmean(image_np, axis=(1, 2), keepdims=True)
    std = np.nanstd(image_np, axis=(1, 2), keepdims=True)
    mean = np.nan_to_num(mean, nan=0.0)
    std = np.nan_to_num(std, nan=1.0)
    normalized_image = (image_np - mean) / (std + 1e-7)
    normalized_image = np.clip(normalized_image, -5.0, 5.0) # Clipping
    return normalized_image

# --- Prediction Dataset Class (Copied from training script) ---
class NpyPredictionDataset(Dataset):
    """ Dataset for loading .npy files for prediction. """
    def __init__(self, root_dir: str, transform: Optional[Callable] = None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.file_paths = sorted([p for p in self.root_dir.glob('*.npy')])
        if not self.file_paths:
            raise FileNotFoundError(f"No .npy files found in {self.root_dir}")
        print(f"Found {len(self.file_paths)} .npy files for prediction in {self.root_dir}.")

    def __len__(self):
        return len(self.file_paths)

    # Return image tensor and path string
    def __getitem__(self, idx: int) -> Optional[Tuple[torch.Tensor, str]]:
        image_path = str(self.file_paths[idx])
        try:
            image_np = load_sentinel2_image(image_path)
            if image_np is None: return None
            image_np_normalized = normalize_image_per_image(image_np)
            if image_np_normalized is None: return None
            image_tensor = torch.from_numpy(image_np_normalized).float()
            if self.transform:
                image_tensor = self.transform(image_tensor)
            return image_tensor, image_path
        except Exception as e:
            print(f"Error processing prediction image {image_path}:")
            traceback.print_exc()
            return None # Signal error

# --- Custom Collate Function (Copied from training script) ---
def collate_fn(batch):
    """ Filters out None samples and stacks the rest. """
    batch = [item for item in batch if item is not None]
    if not batch:
        return None # Return None if the whole batch was invalid
    try:
        # Use default_collate for valid items
        return torch.utils.data.dataloader.default_collate(batch)
    except Exception as e:
        print(f"Error in collate_fn during stacking: {e}. Skipping batch.")
        return None

# --- Transforms (Copied - only need evaluation transforms) ---
img_size = EVAL_CONFIG["data"]["image_size"]
eval_transforms = transforms.Compose([
    transforms.Resize((img_size, img_size), antialias=True),
    # Normalization is done per-image in the Dataset __getitem__
])

# --- Model Architecture Definition ---
# Mish Activation Function (Compatible with training script)
class Mish(nn.Module):
    """Applies the Mish activation function element-wise."""
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

# Squeeze-and-Excitation Block (Compatible with training script)
class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            Mish(),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        squeezed = self.squeeze(x)
        excited = self.excitation(squeezed)
        return x * excited

# --- Sentinel2Classifier Model (Copied from the training script) ---
class Sentinel2Classifier(nn.Module):
    """ Custom CNN for Sentinel-2 image classification with Residual Connections,
        SE Attention, Hybrid Pooling & Mish Activation. """
    def __init__(self, num_classes: int, input_channels: int = 12):
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")

        self.mish = Mish()

        # --- Convolutional Blocks ---
        def conv_block(in_c, out_c, kernel_size=3, stride=1, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(out_c),
                Mish()
            )

        self.conv1 = conv_block(input_channels, 64)
        self.res_conv1 = nn.Conv2d(input_channels, 64, kernel_size=1, bias=False)
        self.se1 = SEBlock(64)

        self.conv2 = conv_block(64, 128)
        self.res_conv2 = nn.Conv2d(64, 128, kernel_size=1, bias=False)
        self.se2 = SEBlock(128)

        self.conv3 = conv_block(128, 256)
        self.res_conv3 = nn.Conv2d(128, 256, kernel_size=1, bias=False)
        self.se3 = SEBlock(256)

        self.conv4 = conv_block(256, 512)
        self.res_conv4 = nn.Conv2d(256, 512, kernel_size=1, bias=False)
        self.se4 = SEBlock(512)

        # --- Pooling ---
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        pooled_features = 512

        # --- Fully Connected Layers ---
        self.fc1 = nn.Linear(pooled_features, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(256, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.4)

        self.fc_out = nn.Linear(128, num_classes)

        # No need to initialize weights here, we are loading them
        # self.apply(self._initialize_weights)
        print(f"Instantiated {self.__class__.__name__} for evaluation.")

    # _initialize_weights method is not needed for evaluation script

    def forward(self, x):
        res1 = self.res_conv1(x)
        x = self.conv1(x)
        x = self.se1(x) + res1

        res2 = self.res_conv2(x)
        x = self.conv2(x)
        x = self.se2(x) + res2

        res3 = self.res_conv3(x)
        x = self.conv3(x)
        x = self.se3(x) + res3

        res4 = self.res_conv4(x)
        x = self.conv4(x)
        x = self.se4(x) + res4

        avg_p = self.avg_pool(x)
        max_p = self.max_pool(x)
        x = avg_p + max_p # Combine pooling outputs
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = self.mish(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = self.mish(x)
        x = self.dropout2(x)

        x = self.fc_out(x)
        return x

# --- Main Evaluation Function ---
def evaluate_model(config: Dict[str, Any], device: torch.device, num_classes: int, class_names: List[str]):
    """Loads model, runs prediction, saves CSV, and optionally submits."""

    # --- Data ---
    try:
        pred_dataset = NpyPredictionDataset(config["data"]["prediction_dir"], transform=eval_transforms)
        pred_loader = DataLoader(
            pred_dataset,
            batch_size=config["data"]["batch_size"],
            shuffle=False,
            num_workers=config["data"]["num_workers"],
            pin_memory=True, # Use pin_memory if using GPU
            collate_fn=collate_fn # Use the safe collate function
        )
    except FileNotFoundError as e:
        print(f"Error creating prediction dataset: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred during data loading setup: {e}")
        traceback.print_exc()
        return

    # --- Model ---
    # !!! Instantiate the CORRECT model class !!!
    model = Sentinel2Classifier(num_classes=num_classes)
    model_path = config["model"]["load_path"]

    if not model_path.exists():
        print(f"ERROR: Model weights file not found at {model_path}")
        return

    try:
        print(f"Loading model weights from: {model_path}")
        # Load state dict - ensure map_location is set for cross-device compatibility
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval() # Set model to evaluation mode
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        traceback.print_exc()
        return

    # --- Prediction ---
    predictions = []
    image_ids = [] # Store image IDs (e.g., filenames)

    print("\n--- Starting Prediction ---")
    with torch.no_grad(): # Disable gradient calculations for evaluation
        progress_bar = tqdm(pred_loader, desc="Predicting")
        for batch_data in progress_bar:
            if batch_data is None:
                print("Warning: Skipping None batch (likely due to loading error).")
                continue # Skip failed batches

            try:
                inputs, paths = batch_data # Unpack image tensors and paths
                inputs = inputs.to(device, non_blocking=True) # Use non_blocking with pin_memory

                # Use autocast context manager for AMP if enabled
                with autocast(dtype=torch.float16, enabled=AMP_ENABLED):
                    outputs = model(inputs)

                # Get predicted class indices (highest logit score)
                _, predicted_indices = torch.max(outputs, 1)
                predictions.extend(predicted_indices.cpu().numpy())

                # Extract image IDs (e.g., filename without extension)
                batch_ids = [Path(p).stem for p in paths] # Get 'test_0000' etc.
                image_ids.extend(batch_ids)

            except Exception as e:
                print(f"\nError during prediction batch: {e}")
                traceback.print_exc()
                # Decide whether to stop or continue
                # continue

    if not predictions:
        print("\nError: No predictions were generated. Check dataset and prediction loop.")
        return
    if not image_ids:
        print("\nError: No image IDs were collected. Check dataset and prediction loop.")
        return

    # --- Generate Submission File ---
    print(f"\nGenerated {len(predictions)} predictions for {len(image_ids)} images.")
    if len(predictions) != len(image_ids):
         # Handle potential mismatch if some images failed loading in batches
         print(f"Warning: Mismatch between predictions ({len(predictions)}) and image IDs ({len(image_ids)}). Check for errors during data loading/prediction.")
         # Attempt to proceed with the minimum count, but investigate the cause
         min_len = min(len(predictions), len(image_ids))
         predictions = predictions[:min_len]
         image_ids = image_ids[:min_len]
         print(f"Proceeding with {min_len} pairs.")


    try:
        # Map predicted indices to class names
        predicted_class_names = [class_names[idx] for idx in predictions]

        pred_df = pd.DataFrame({
            'test_id': image_ids,
            'label': predicted_class_names
        })

        # Optional: Clean test_id if needed (e.g., remove 'test_')
        # pred_df['test_id'] = pred_df['test_id'].str.replace('test_', '', regex=False)

        csv_path = config["prediction"]["predictions_csv_path"]
        # Ensure output directory exists
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(csv_path, index=False)
        print(f"Predictions saved to: {csv_path}")

    except IndexError as e:
         print(f"\nError mapping prediction indices to class names: {e}")
         print(f"Predicted index {e} is out of bounds for class_names list (length {len(class_names)}).")
         print("Ensure NUM_CLASSES and CLASS_NAMES in EVAL_CONFIG match the trained model exactly.")
         return
    except Exception as e:
         print(f"\nError creating or saving submission CSV: {e}")
         traceback.print_exc()
         return


    # --- Kaggle Submission (Optional) ---
    if config["prediction"]["submit_to_kaggle"]:
        print("\n--- Attempting Kaggle Submission ---")
        try:
            # Ensure kaggle package is installed: pip install kaggle
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate() # Requires kaggle.json to be set up

            competition = config["prediction"]["kaggle_competition"]
            message = config["prediction"]["kaggle_message"]

            if not competition:
                 print("Kaggle competition slug is not set in EVAL_CONFIG. Skipping submission.")
                 return

            print(f"Submitting {csv_path} to competition: {competition}")
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
            print("Check kaggle.json configuration, competition slug, and API key validity.")

# --- Run Evaluation ---
if __name__ == '__main__':
    # Ensure output directory exists before starting
    try:
        EVAL_CONFIG["prediction"]["predictions_csv_path"].parent.mkdir(parents=True, exist_ok=True)
        evaluate_model(EVAL_CONFIG, DEVICE, NUM_CLASSES, CLASS_NAMES)
    except Exception as main_e:
        print(f"\nA critical error occurred during script execution: {main_e}")
        traceback.print_exc()

    print("\nEvaluation script finished.")
