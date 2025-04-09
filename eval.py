import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
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
# Use standard tqdm
from tqdm import tqdm
from typing import Tuple, List, Dict, Any, Optional, Callable

# --- Configuration (Adjust as needed) ---
EVAL_CONFIG = {
    "model": {
        "load_path": Path('./outputs/sentinel2.pth'),
        # !!! IMPORTANT: Set these based on your training output !!!
        "num_classes": 10, # e.g., 10
        "class_names": ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake'], # e.g., ['class1', 'class2', ...]
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
        "predictions_csv_path": Path('./outputs/_track1.csv'), # Output file
        "kaggle_competition": 'aicrowd-geospatial-challenge', # Replace if needed
        "kaggle_message": 'Evaluation Script Submission',
        "submit_to_kaggle": False, # Set to False to skip submission attempt
    }
}

# --- Basic Setup ---
DEVICE = torch.device(EVAL_CONFIG["device"])
AMP_ENABLED = EVAL_CONFIG["amp_enabled"] and DEVICE.type == 'cuda'
print(f"Using device: {DEVICE}")
print(f"Automatic Mixed Precision (AMP) enabled: {AMP_ENABLED}")

# --- Make sure NUM_CLASSES and CLASS_NAMES are set ---
if EVAL_CONFIG["model"]["num_classes"] is None or EVAL_CONFIG["model"]["class_names"] is None:
    raise ValueError("Please set 'num_classes' and 'class_names' in EVAL_CONFIG['model'] based on your training run.")
NUM_CLASSES = EVAL_CONFIG["model"]["num_classes"]
CLASS_NAMES = EVAL_CONFIG["model"]["class_names"]
print(f"Expecting {NUM_CLASSES} classes: {CLASS_NAMES}")


# --- Data Loading Function (MODIFIED FOR NPY) ---
def load_sentinel2_image(filepath: Path) -> np.ndarray:
    """Loads a Sentinel-2 image (TIF or NPY), returns NumPy CHW (12 bands)."""
    filepath_str = str(filepath)

    if filepath.suffix.lower() in ['.tif', '.tiff']:
        # This part remains the same, in case TIF files are encountered
        try:
            with rasterio.open(filepath_str) as src:
                bands_to_read = list(range(1, 10)) + list(range(11, 14)) # 12 bands
                if src.count < max(bands_to_read):
                     raise ValueError(f"Expected at least {max(bands_to_read)} bands, got {src.count} in {filepath}")
                image = src.read(bands_to_read)
        except rasterio.RasterioIOError as e:
            raise IOError(f"Error reading TIF file {filepath}: {e}")

    elif filepath.suffix.lower() == '.npy':
        # --- MODIFIED NPY LOADING ---
        try:
            image = np.load(filepath_str) # Load NPY
            #print(f"Loaded NPY: {filepath.name}, Original Shape: {image.shape}") # Debug print

            # Check shape and transpose if necessary (H, W, C) -> (C, H, W)
            if image.ndim == 3 and image.shape[2] == 12:
                #print(f"Transposing NPY from (H, W, C) to (C, H, W)")
                image = np.transpose(image, (2, 0, 1)) # Transpose HWC to CHW
            elif image.ndim == 3 and image.shape[0] == 12:
                 # Already in (C, H, W) format, do nothing extra
                 #print(f"NPY already in (C, H, W) format.")
                 pass
            else:
                 # If it's not (H, W, 12) or (12, H, W), raise error
                 raise ValueError(f"Unexpected shape for .npy {filepath}: {image.shape}. Expected (H, W, 12) or (12, H, W).")

            #print(f"Processed NPY Shape: {image.shape}") # Debug print

        except Exception as e:
            # Catch potential loading errors or shape errors from above
            raise IOError(f"Error loading or processing NPY file {filepath}: {e}")
        # --- END OF MODIFICATION ---
    else:
        raise ValueError(f"Unsupported file type: {filepath}")

    # Ensure float32 type
    return image.astype(np.float32)

# --- Per-Image Normalization Function (Copied) ---
def normalize_image_per_image(image_np: np.ndarray) -> np.ndarray:
    """Normalizes 12-channel NumPy image (C, H, W) using its own stats."""
    if image_np.ndim != 3 or image_np.shape[0] != 12:
        # This check should pass after the transpose in load_sentinel2_image
        raise ValueError(f"Invalid shape for per-image normalization: {image_np.shape}. Expected (12, H, W).")

    mean = np.mean(image_np, axis=(1, 2), keepdims=True)
    std = np.std(image_np, axis=(1, 2), keepdims=True)
    std[std == 0] = 1e-7 # Avoid division by zero

    return (image_np - mean) / std

# --- Prediction Dataset Class (Copied) ---
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
            # Use the modified load function
            image_np = load_sentinel2_image(image_path)
            image_np = normalize_image_per_image(image_np) # Apply per-image normalization
            image_tensor = torch.from_numpy(image_np.copy()).float() # Convert to tensor

            if self.transform:
                image_tensor = self.transform(image_tensor) # Apply resizing

            # Return image tensor and its original path
            return image_tensor, str(image_path)

        except Exception as e:
            print(f"ERROR loading/processing prediction image {image_path}: {e}\n{traceback.format_exc()}")
            return None # Signal error

# --- Custom Collate Function (Copied) ---
def safe_collate(batch: List[Optional[Any]]) -> Optional[List[Any]]:
    """Collate function that filters out None results from failed __getitem__ calls."""
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

# --- Transforms (Copied - only need evaluation transforms) ---
img_size = EVAL_CONFIG["data"]["image_size"]
eval_transforms = transforms.Compose([
    transforms.Resize((img_size, img_size), antialias=True),
    # Normalization is done per-image in the Dataset __getitem__
])

# --- Model Architecture (Copied - needed to load state_dict) ---
class Mish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(F.softplus(x))

class SEBlock(nn.Module):
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
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1, use_se: bool = True):
        super().__init__()
        self.use_se = use_se
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
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
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = BasicConvBlock(channels, channels, use_se=False)
        self.conv2 = BasicConvBlock(channels, channels, use_se=False)
        self.se = SEBlock(channels)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.se(out + residual)
        return out

class Sentinel2Classifier_v2(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 12):
        super().__init__()
        self.stem = BasicConvBlock(in_channels, 64, use_se=True)
        self.layer1 = self._make_layer(64, 128, num_blocks=2)
        self.layer2 = self._make_layer(128, 256, num_blocks=2)
        self.layer3 = self._make_layer(256, 512, num_blocks=2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        feature_dim = 512
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(feature_dim * 2, 256),
            Mish(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
        # No need to initialize weights here, we are loading them
        # self.apply(self._initialize_weights)
        print(f"Instantiated {self.__class__.__name__} for evaluation.")

    def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int) -> nn.Sequential:
        layers = []
        layers.append(BasicConvBlock(in_channels, out_channels, stride=2, use_se=True))
        for _ in range(num_blocks):
            layers.append(ResBlock(out_channels))
        return nn.Sequential(*layers)

    # _initialize_weights is not needed for evaluation script

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.head(x)
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
            pin_memory=True,
            collate_fn=safe_collate
        )
    except FileNotFoundError as e:
        print(f"Error creating prediction dataset: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred during data loading setup: {e}")
        return

    # --- Model ---
    model = Sentinel2Classifier_v2(num_classes=num_classes)
    model_path = config["model"]["load_path"]

    if not model_path.exists():
        print(f"ERROR: Model weights file not found at {model_path}")
        return

    try:
        print(f"Loading model weights from: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval() # Set model to evaluation mode
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        return

    # --- Prediction ---
    predictions = []
    image_ids = []
    scaler = GradScaler(enabled=AMP_ENABLED) # AMP scaler

    print("\n--- Starting Prediction ---")
    with torch.no_grad(): # Disable gradient calculations
        progress_bar = tqdm(pred_loader, desc="Predicting")
        for batch_data in progress_bar:
            if batch_data is None:
                print("Warning: Skipping None batch.")
                continue # Skip failed batches

            try:
                inputs, paths = batch_data # Unpack image tensors and paths
                inputs = inputs.to(device)

                # Use autocast context manager for AMP
                with autocast(device_type=device.type, enabled=scaler.is_enabled()):
                    outputs = model(inputs)

                # Get predicted class indices
                _, predicted_indices = torch.max(outputs, 1)
                predictions.extend(predicted_indices.cpu().numpy())

                # Extract image IDs (e.g., filename without extension)
                batch_ids = [Path(p).stem for p in paths] # Get 'test_0000' etc.
                image_ids.extend(batch_ids)

            except Exception as e:
                print(f"\nError during prediction batch: {e}")
                # Optionally continue to next batch or stop
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
         print(f"Warning: Mismatch between predictions ({len(predictions)}) and image IDs ({len(image_ids)}). Using {min(len(predictions), len(image_ids))} pairs.")
         min_len = min(len(predictions), len(image_ids))
         predictions = predictions[:min_len]
         image_ids = image_ids[:min_len]

    try:
        # Map predicted indices to class names
        predicted_class_names = [class_names[idx] for idx in predictions]

        pred_df = pd.DataFrame({
            'test_id': image_ids,
            'label': predicted_class_names
        })

        # Optional: Clean test_id if needed (e.g., remove 'test_')
        pred_df['test_id'] = pred_df['test_id'].str.replace('test_', '', regex=False)

        csv_path = config["prediction"]["predictions_csv_path"]
        # Ensure output directory exists
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(csv_path, index=False)
        print(f"Predictions saved to: {csv_path}")

    except IndexError as e:
         print(f"\nError mapping prediction indices to class names: {e}")
         print("Ensure NUM_CLASSES and CLASS_NAMES in EVAL_CONFIG match the trained model.")
         return
    except Exception as e:
         print(f"\nError creating or saving submission CSV: {e}")
         return


    # --- Kaggle Submission (Optional) ---
    if config["prediction"]["submit_to_kaggle"]:
        print("\n--- Attempting Kaggle Submission ---")
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()

            competition = config["prediction"]["kaggle_competition"]
            message = config["prediction"]["kaggle_message"]

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
            print("Check kaggle.json configuration and competition slug.")

# --- Run Evaluation ---
if __name__ == '__main__':
    evaluate_model(EVAL_CONFIG, DEVICE, NUM_CLASSES, CLASS_NAMES)
    print("\nEvaluation script finished.")

