import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import numpy as np
import pandas as pd
import os
import torchvision
import rasterio
import random
from torch.cuda.amp import autocast
from torchvision.models import ResNet50_Weights, ConvNeXt_Base_Weights # Keep relevant imports
import traceback

print(f"PyTorch Version: {torch.__version__}")
print(f"Torchvision Version: {torchvision.__version__}") # Use transforms for version

# --- Configuration ---
model_save_path = 'convnext_staged_imgnorm.pth' # Path to the trained model
test_data_dir = './testset/testset'           # Directory with .npy test files
output_csv_path = 'track_2.csv'               # Output CSV filename
kaggle_competition = '8-860-1-00-coding-challenge-2025' # Your competition slug
kaggle_message = 'ConvNeXt Staged - Per Image Norm Eval' # Submission message

BATCH_SIZE = 64 # Adjust based on GPU memory during inference
NUM_WORKERS = 4
IMAGE_SIZE = 256 # Must match the input size the model was trained on
NUM_CLASSES = 10 # CRITICAL: Set to the correct number of classes
# CRITICAL: Define class names in the exact order
CLASS_NAMES = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
    'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake'
]
if len(CLASS_NAMES) != NUM_CLASSES:
    raise ValueError(f"Length of CLASS_NAMES ({len(CLASS_NAMES)}) must match NUM_CLASSES ({NUM_CLASSES})")

# --- Device Setup ---
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
print(f"Using device: {DEVICE}")

# --- REMOVED Global Normalization Stats ---

# --- Data Loading Function ---
def load_sentinel2_image(filepath):
    """Loads a Sentinel-2 image (12 bands) from .npy, returns NumPy CHW."""
    if filepath.endswith('.npy'):
        image = np.load(filepath)
        if image.shape[0] == 13: # Handle 13-band images if present
            image = np.concatenate((image[:9], image[10:]), axis=0) # Skip band 10 -> (C, H, W)
        # Ensure CHW format if loaded as HWC
        if len(image.shape) == 3 and image.shape[0] != 12 and image.shape[2] == 12:
            image = image.transpose(2, 0, 1) # HWC -> CHW
        elif image.shape[0] != 12:
             raise ValueError(f"Unexpected shape for .npy {filepath}: {image.shape}")
    else:
        raise ValueError(f"Unsupported file type for prediction: {filepath}. Expected .npy")
    return image.astype(np.float32) # Ensure float32

# --- Per-Image Normalization Function ---
def normalize_image_per_image(image_np):
    """Normalizes 12-channel NumPy image (C, H, W) using its own stats."""
    if image_np.ndim != 3 or image_np.shape[0] != 12:
         raise ValueError(f"Invalid shape for per-image normalization: {image_np.shape}. Expected (12, H, W).")
    mean = np.mean(image_np, axis=(1, 2), keepdims=True)
    std = np.std(image_np, axis=(1, 2), keepdims=True)
    # Add epsilon inside the denominator before division
    normalized_image = (image_np - mean) / (std + 1e-7)
    return normalized_image # Returns NumPy array

# --- Prediction Dataset (Using Per-Image Normalization) ---
class NpyPredictionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_paths = sorted([os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.npy')])
        if not self.file_paths:
            raise FileNotFoundError(f"No .npy files found in directory: {root_dir}")
        print(f"Found {len(self.file_paths)} .npy files for prediction.")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image_path = self.file_paths[idx]
        try:
            image_np = load_sentinel2_image(image_path) # NumPy (C, H, W)
            image_np = normalize_image_per_image(image_np) # Apply per-image normalization HERE

            image_tensor = torch.from_numpy(image_np).float() # Convert to tensor AFTER normalization

            if self.transform:
                image_tensor = self.transform(image_tensor) # Apply remaining transforms (Resize)

            return image_tensor, 0, image_path # Dummy label 0, return path

        except Exception as e:
            print(f"Error loading/processing image {image_path}:")
            traceback.print_exc()
            return None, None, None

# --- Prediction Transforms (REVISED - NO Normalize Step) ---
pred_transforms = transforms.Compose([
    # Input is already a normalized 12-channel Tensor from __getitem__
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True), # Resize
    # NO Normalize step here
])

# --- Model Definition (MUST MATCH SAVED MODEL) ---
# Using ConvNeXt adapted for 12->3 channels internally
class ConvNeXtSentinel2(nn.Module):
    """ ConvNeXt model adapted for 12-channel input using an initial 1x1 conv. """
    def __init__(self, num_classes, pretrained=False): # Set pretrained=False for loading
        super().__init__()
        self.channel_projector = nn.Conv2d(12, 3, kernel_size=1, bias=False)
        self.convnext_base = models.convnext_base(weights=None) # Load structure only

        # Rebuild the classifier head structure EXACTLY as during training
        num_features_in = self.convnext_base.classifier[-1].in_features
        self.convnext_base.classifier = nn.Sequential(
             self.convnext_base.classifier[0], # Keep LayerNorm
             self.convnext_base.classifier[1], # Keep Flatten
             nn.Dropout(p=0.5),                # Keep dropout
             nn.Linear(num_features_in, num_classes) # Final Linear layer
        )
        # No weight initialization needed - state dict will overwrite

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_projector(x) # 12 -> 3 channels
        x = self.convnext_base(x)     # Pass 3ch tensor to standard ConvNeXt
        return x

# --- Main Execution ---
if __name__ == '__main__':

    # 1. Create Model instance
    print("Creating model architecture...")
    model = ConvNeXtSentinel2(num_classes=NUM_CLASSES, pretrained=False)
    print(f"Model: {type(model).__name__}")

    # 2. Load Saved State Dict
    print(f"Loading model weights from: {model_save_path}")
    try:
        model.load_state_dict(torch.load(model_save_path, map_location=DEVICE, weights_only=True)) # Safe loading
        print("Model weights loaded successfully.")
    except FileNotFoundError:
        print(f"ERROR: Model weights file not found at {model_save_path}")
        exit()
    except RuntimeError as e:
        print(f"ERROR: Failed to load state_dict. Architecture mismatch likely.")
        print("Ensure the ConvNeXtSentinel2 class definition here matches the one used for training.")
        print(e)
        exit()
    except Exception as e:
        print(f"An unexpected error occurred during model loading:")
        traceback.print_exc()
        exit()

    model.to(DEVICE) # Move model to GPU/CPU
    model.eval()     # Set to evaluation mode

    # 3. Create Prediction Dataset and DataLoader
    print(f"Loading prediction data from: {test_data_dir}")
    pred_dataset = NpyPredictionDataset(test_data_dir, transform=pred_transforms) # Apply transforms
    def pred_collate_fn(batch): # Simplified collate for prediction
        batch = list(filter(lambda x: x is not None and x[0] is not None, batch))
        if not batch: return None, None
        try:
            images = torch.stack([item[0] for item in batch])
            paths = [item[2] for item in batch] # Get paths from dataset item[2]
            return images, paths
        except Exception as e:
             print(f"Error in prediction collate_fn: {e}")
             traceback.print_exc()
             return None, None


    pred_loader = DataLoader(pred_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True, collate_fn=pred_collate_fn)

    # 4. Perform Prediction
    print("Starting prediction loop...")
    all_predictions = []
    all_filenames_base = [] # Store filenames without extension

    with torch.no_grad(): # Essential for inference
        for batch_data in pred_loader:
            if batch_data is None or batch_data[0] is None:
                print("Warning: Skipping empty/invalid prediction batch")
                continue
            inputs, image_paths_batch = batch_data # Unpack image tensor and list of paths

            inputs = inputs.to(DEVICE)
            with torch.amp.autocast(device_type=DEVICE.type, enabled=USE_CUDA): # Use autocast
                outputs = model(inputs)

            _, predicted_indices = torch.max(outputs, 1) # Get class indices

            # Map indices to class names
            predicted_names = [CLASS_NAMES[idx.item()] for idx in predicted_indices]
            all_predictions.extend(predicted_names)

            # Extract just the filename without extension for test_id
            all_filenames_base.extend([os.path.splitext(os.path.basename(path))[0] for path in image_paths_batch])

    print("Prediction loop finished.")

    # 5. Create and Save Submission CSV
    if not all_predictions or not all_filenames_base:
        print("ERROR: No predictions were generated. Check dataset and prediction loop.")
    elif len(all_predictions) != len(all_filenames_base):
         print(f"ERROR: Mismatch between prediction count ({len(all_predictions)}) and filename count ({len(all_filenames_base)}).")
    else:
        print(f"Generated {len(all_predictions)} predictions.")
        predictions_df = pd.DataFrame({
            'test_id': all_filenames_base, # Use the already processed names
            'label': all_predictions
        })
        try:
            # Convert test_id to integer, removing 'test_' prefix if present
            predictions_df['test_id'] = predictions_df['test_id'].str.replace('test_', '', regex=False).astype(int)
        except ValueError:
            print("Warning: Could not convert all test_ids to integers after removing prefix. Saving as strings.")
        except Exception as e:
             print(f"Warning: Error during test_id conversion: {e}. Saving as strings.")


        predictions_df.to_csv(output_csv_path, index=False)
        print(f"Submission predictions saved to: {output_csv_path}")

    print("\nScript finished.")