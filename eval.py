import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torchvision.transforms.functional import hflip # For TTA
import numpy as np
import pandas as pd
import os
import time
import traceback
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, List, Dict, Any, Optional, Callable
import warnings
# YAML and Argparse Imports
import yaml # <<< Added
import argparse # <<< Added

# --- Argument Parser ---
parser = argparse.ArgumentParser(description='Evaluate ResNet model for EuroSAT classification.')
parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file for evaluation.')
args = parser.parse_args()

# --- Load Configuration ---
config_path = Path(args.config)
if not config_path.is_file():
    raise FileNotFoundError(f"Configuration file not found: {config_path}")
try:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f) # <<< Load config from YAML
    print(f"Loaded evaluation configuration from: {config_path}")
except yaml.YAMLError as e:
    raise ValueError(f"Error parsing YAML file: {e}")
except Exception as e:
    raise IOError(f"Error reading config file: {e}")


# --- Device Setup (Override config if needed) ---
USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    DEVICE = torch.device("cuda")
    config['device'] = 'cuda' # Ensure config reflects actual device
    config['amp_enabled'] = True # Default AMP True if CUDA available
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    config['device'] = 'mps'
    config['amp_enabled'] = False
else:
    DEVICE = torch.device("cpu")
    config['device'] = 'cpu'
    config['amp_enabled'] = False
print(f"Using device: {DEVICE}")
# Use AMP setting determined by device availability unless explicitly set to False in config
AMP_ENABLED = config.get('amp_enabled', USE_CUDA) # <<< Use loaded config, default based on CUDA
print(f"Automatic Mixed Precision (AMP) enabled for evaluation: {AMP_ENABLED}")


# --- Suppress warnings ---
warnings.filterwarnings("ignore")

# --- Validate Config (Num Classes / Class Names) ---
# <<< Get class info directly from loaded config
NUM_CLASSES = config["model"].get("num_classes")
CLASS_NAMES = config["model"].get("class_names")
if NUM_CLASSES is None or CLASS_NAMES is None:
    raise ValueError("Config must contain 'model.num_classes' and 'model.class_names'")
if len(CLASS_NAMES) != NUM_CLASSES:
     raise ValueError(f"Config Mismatch: num_classes ({NUM_CLASSES}) != length of class_names ({len(CLASS_NAMES)})")
print(f"Expecting {NUM_CLASSES} classes: {CLASS_NAMES}")


# --- Data Loading and Preprocessing (Matching Training Script) ---

# --- Band Mapping (Standard Order) ---
BAND_MAP_12 = {
    "B1_Coastal": 0, "B2_Blue": 1, "B3_Green": 2, "B4_Red": 3,
    "B5_RE1": 4, "B6_RE2": 5, "B7_RE3": 6, "B8_NIR": 7,
    "B8A_NIR2": 8, "B9_WV": 9, "B11_SWIR1": 10, "B12_SWIR2": 11
}
print(f"Using BAND_MAP_12 reflecting standard order: {BAND_MAP_12}")

def load_npy_image(filepath: str) -> Optional[np.ndarray]:
    """Loads NPY file, assuming 12 channels (C, H, W) or (H, W, C), returns (12, H, W) NumPy array or None."""
    try:
        filepath_lower = filepath.lower(); image_data = None
        if not filepath_lower.endswith('.npy'): return None
        image = np.load(filepath)
        if image.ndim == 3 and image.shape[0] != 12 and image.shape[2] == 12: image = image.transpose(2, 0, 1)
        elif image.ndim != 3 or image.shape[0] != 12: print(f"Error: NPY Shape {filepath}: {image.shape}"); return None
        return image.astype(np.float32)
    except Exception as e: print(f"Error loading NPY {filepath}: {e}"); traceback.print_exc(); return None

# --- Index Calculation ---
def calculate_indices_from_raw(image_np_12bands: np.ndarray, epsilon=1e-7) -> Dict[str, np.ndarray]:
    indices = {}; clip_val = 1.0; global BAND_MAP_12
    try:
        # (Index calculation logic remains the same)
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

# --- Prediction Dataset Class ---
class NpyPredictionDataset(Dataset):
    """ Dataset: Loads NPY (12ch), calculates indices (raw), stacks (16ch), normalizes (16ch), transforms. """
    def __init__(self, root_dir: str, transform: Optional[Callable] = None, output_channels: int = 16): # <<< Pass output_channels
        self.root_dir = Path(root_dir); self.transform = transform
        self.output_channels = output_channels # <<< Use passed value
        self.file_paths = sorted([p for p in self.root_dir.glob('*.npy')])
        if not self.file_paths: raise FileNotFoundError(f"No .npy files found in {self.root_dir}")
        print(f"Initialized NpyPredictionDataset with {len(self.file_paths)} files. Output channels: {self.output_channels}")
    def __len__(self): return len(self.file_paths)
    def __getitem__(self, idx: int) -> Optional[Tuple[torch.Tensor, str]]:
        image_path = str(self.file_paths[idx])
        try:
            image_np_12 = load_npy_image(image_path);
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
            return image_tensor, image_path
        except Exception as e: print(f"Error processing {image_path}: {e}"); return None

# --- Custom Collate Function ---
def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch: return None
    try: return torch.utils.data.dataloader.default_collate(batch)
    except Exception as e: print(f"Error in collate_fn: {e}"); return None

# --- Transforms (Evaluation) ---
# <<< Use loaded config
IMG_SIZE = config["data"]["image_size"]
eval_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
])

# --- Model Definition Function ---
# <<< Use the same function definition as in the training script
def adapt_resnet_for_multichannel(model_name="resnet50", pretrained=False, num_classes=10, input_channels=16):
    """Loads ResNet architecture, optionally loads pretrained weights, adapts layers."""
    print(f"Loading {model_name} architecture definition...")
    model = models.get_model(model_name, weights=None) # Always load architecture only
    original_conv1=model.conv1; new_conv1=nn.Conv2d(input_channels, original_conv1.out_channels, kernel_size=original_conv1.kernel_size, stride=original_conv1.stride, padding=original_conv1.padding, bias=original_conv1.bias is not None)
    model.conv1 = new_conv1; print(f"Adapted model.conv1 architecture for {input_channels} inputs.")
    num_ftrs = model.fc.in_features; model.fc = nn.Linear(num_ftrs, num_classes); print(f"Adapted model.fc architecture for {num_classes} outputs.")
    # No random init needed here, weights loaded from state_dict later
    return model

# --- Main Evaluation Function (with TTA Option) ---
def evaluate_model(config: Dict[str, Any], device: torch.device, num_classes: int, class_names: List[str]):
    """Loads adapted ResNet model, runs prediction (with optional TTA), saves CSV."""
    # <<< Use loaded config
    use_tta = config["prediction"].get("use_tta", False)
    image_size = config["data"]["image_size"]
    input_channels = config["model"]["input_channels"]
    # Convert paths from config strings to Path objects
    prediction_dir = Path(config["data"]["prediction_dir"])
    base_save_path = Path(config["model"]["base_save_path"])
    load_weights_name = config["model"]["load_weights_name"]
    model_load_path = base_save_path / load_weights_name # Construct full path
    predictions_csv_path = Path(config["prediction"]["predictions_csv_path"])


    # --- Data ---
    try:
        pred_dataset = NpyPredictionDataset(
            root_dir=str(prediction_dir), # NpyPredictionDataset expects string path
            transform=eval_transforms,
            output_channels=input_channels # Pass from config
        )
        pred_loader = DataLoader( pred_dataset, batch_size=config["data"]["batch_size"], shuffle=False,
            num_workers=config["data"]["num_workers"], pin_memory=True, collate_fn=collate_fn )
    except Exception as e: print(f"Error creating prediction dataset/loader: {e}"); return

    # --- Model ---
    model = adapt_resnet_for_multichannel(
        model_name=config["model"].get("architecture", "resnet50"), # Allow specifying arch in config
        pretrained=config["model"]["pretrained"], # Pass flag from config
        num_classes=num_classes,
        input_channels=input_channels
    )

    if not model_load_path.exists(): print(f"ERROR: Model weights file not found at {model_load_path}"); return
    try:
        print(f"Loading model weights from: {model_load_path}")
        state_dict = torch.load(model_load_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device); model.eval()
        print("Model loaded successfully.")
    except Exception as e: print(f"Error loading model state_dict: {e}"); return

    # --- Prediction ---
    all_final_preds = []; image_ids = []
    tta_mode_str = "with TTA (Orig+HFlip)" if use_tta else "without TTA"
    print(f"\n--- Starting Prediction {tta_mode_str} ---")

    with torch.no_grad():
        progress_bar = tqdm(pred_loader, desc=f"Predicting {tta_mode_str}")
        for batch_data in progress_bar:
            # (Prediction loop with TTA logic remains the same)
            if batch_data is None: continue
            try:
                inputs, paths = batch_data
                inputs = inputs.to(device, non_blocking=True)
                batch_ids = [Path(p).stem.replace('test_', '') for p in paths]
                image_ids.extend(batch_ids)
                with autocast(dtype=torch.float16, enabled=AMP_ENABLED):
                    outputs_orig = model(inputs)
                    if use_tta:
                        inputs_hflip = hflip(inputs)
                        outputs_hflip = model(inputs_hflip)
                        probs_orig = F.softmax(outputs_orig, dim=1)
                        probs_hflip = F.softmax(outputs_hflip, dim=1)
                        avg_probs = (probs_orig + probs_hflip) / 2.0
                        _, predicted_indices = torch.max(avg_probs, 1)
                    else:
                        _, predicted_indices = torch.max(outputs_orig, 1)
                all_final_preds.extend(predicted_indices.cpu().numpy())
            except Exception as e: print(f"\nError during prediction batch: {e}")

    # --- Generate Submission File ---
    predictions = all_final_preds
    print(f"\nGenerated {len(predictions)} predictions for {len(image_ids)} images ({tta_mode_str}).")
    if not predictions or not image_ids: print("\nError: No predictions generated."); return
    if len(predictions)!=len(image_ids): print(f"Warning: Pred/ID mismatch"); min_len=min(len(predictions),len(image_ids)); predictions=predictions[:min_len]; image_ids=image_ids[:min_len]

    try:
        predicted_class_names = [class_names[idx] for idx in predictions]
        pred_df = pd.DataFrame({'test_id': image_ids, 'label': predicted_class_names})
        csv_path = predictions_csv_path # Use path from config
        if use_tta: csv_path = csv_path.parent / f"{csv_path.stem}_tta{csv_path.suffix}"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(csv_path, index=False)
        print(f"Predictions saved to: {csv_path}")
    except IndexError as e: print(f"\nError mapping prediction index: {e}. Check class_names in config."); return
    except Exception as e: print(f"\nError creating/saving CSV: {e}"); return

    # --- Kaggle Submission ---
    if config["prediction"].get("submit_to_kaggle", False): # Use .get for optional key
        print("\n--- Attempting Kaggle Submission ---")
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi(); api.authenticate()
            competition = config["prediction"].get("kaggle_competition")
            message = config["prediction"].get("kaggle_message", "Submission")
            if not competition: print("Kaggle competition slug not set in config. Skipping."); return
            print(f"Submitting {csv_path} to competition: {competition}")
            api.competition_submit(file_name=str(csv_path), message=message, competition=competition)
            print("Submission successful!")
        except ImportError: print("Kaggle API not found. Install with: pip install kaggle")
        except Exception as e: print(f"Kaggle API submission failed: {e}")

# --- Run Evaluation ---
if __name__ == '__main__':
    try:
        # Ensure output directory exists based on loaded config path
        Path(config["prediction"]["predictions_csv_path"]).parent.mkdir(parents=True, exist_ok=True)
        # Pass loaded config and determined class info
        evaluate_model(config, DEVICE, NUM_CLASSES, CLASS_NAMES)
    except Exception as main_e: print(f"\nCritical error: {main_e}"); traceback.print_exc()
    print("\nEvaluation script finished.")

