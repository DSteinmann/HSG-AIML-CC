import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import models, transforms
import rasterio
import numpy as np
import pandas as pd
import os
import random
import time
from torch.cuda.amp import GradScaler, autocast
from torchvision.models import ResNet50_Weights, ConvNeXt_Base_Weights
from sklearn.model_selection import train_test_split
import traceback

# --- Setup ---
if not os.path.exists('./outputs'):
    os.mkdir('./outputs')

# Define the paths
train_dir = './ds/images/remote_sensing/otherDatasets/sentinel_2/tif'
validation_dir = './testset/testset'  # Test data (.npy files) for final prediction
model_save_path = 'convnext_staged_imgnorm.pth' # New name for this version
predictions_csv_path = 'track_2.csv'

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
print(f"Using device: {DEVICE}")

# --- Configuration & Hyperparameters ---
seed = 1337
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

best_image_size = 256
best_batch_size = 64
best_weight_decay =  3.7553e-05 # Keep from previous attempt
best_optimizer_name = 'AdamW'

# Staged Learning Rates & Epochs (Keep same as before, adjust if needed)
lr_stage1 = 1e-4
lr_stage2 = 3e-5
lr_stage3 = 1e-5
stage1_epochs = 10
stage2_epochs = 20
stage3_epochs = 20
num_epochs = stage1_epochs + stage2_epochs + stage3_epochs # Total max

warmup_epochs = 3
initial_warmup_lr = 1e-6
num_workers = 4
train_ratio = 0.9
patience = 7
# num_classes determined from data scan

# --- REMOVED Global Normalization Stats ---

# --- Data Loading Function ---
def load_sentinel2_image(filepath):
    """Loads a Sentinel-2 image (12 bands), returns NumPy CHW."""
    if filepath.endswith('.tif'):
        with rasterio.open(filepath) as src:
            if src.count < 13: raise ValueError(f"Expected >=13 bands, got {src.count} in {filepath}")
            bands = list(range(1, 10)) + list(range(11, 14))
            image = src.read(bands)
    elif filepath.endswith('.npy'):
        image = np.load(filepath)
        if image.shape[0] == 13:
            image = np.concatenate((image[:9], image[10:]), axis=0)
        if len(image.shape) == 3 and image.shape[0] != 12 and image.shape[2] == 12:
            image = image.transpose(2, 0, 1)
        elif image.shape[0] != 12:
             raise ValueError(f"Unexpected shape for .npy {filepath}: {image.shape}")
    else:
        raise ValueError("Unsupported file type.")
    return image.astype(np.float32)

# --- Per-Image Normalization Function ---
def normalize_image_per_image(image_np):
    """Normalizes 12-channel NumPy image (C, H, W) using its own stats."""
    mean = np.mean(image_np, axis=(1, 2), keepdims=True)
    std = np.std(image_np, axis=(1, 2), keepdims=True)
    normalized_image = (image_np - mean) / (std + 1e-7) # Add epsilon
    return normalized_image # Returns NumPy array

# --- Dataset Class (Using Per-Image Normalization) ---
class Sentinel2Dataset(Dataset):
    """Custom Dataset for Sentinel-2 images. Returns Tensor CHW (12 channels)."""
    def __init__(self, paths_labels, transform=None):
        self.paths_labels = paths_labels
        self.transform = transform
        # Class info handled globally

    def __len__(self):
        return len(self.paths_labels)

    def __getitem__(self, idx):
        image_path, label = self.paths_labels[idx]
        try:
            image_np = load_sentinel2_image(image_path) # NumPy (C, H, W)
            image_np = normalize_image_per_image(image_np) # Apply per-image normalization HERE

            image_tensor = torch.from_numpy(image_np).float() # Convert to tensor
            label_tensor = torch.tensor(label, dtype=torch.long)

            if self.transform:
                image_tensor = self.transform(image_tensor) # Apply remaining transforms

            return image_tensor, label_tensor, image_path

        except Exception as e:
            print(f"Error loading/processing image {image_path}:")
            traceback.print_exc()
            return None, None, None

# Prediction dataset also uses per-image normalization via transforms
class NpyPredictionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_paths = sorted([os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.npy')])
        if not self.file_paths: raise FileNotFoundError(f"No .npy files in {root_dir}")
        print(f"Found {len(self.file_paths)} .npy files for prediction.")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image_path = self.file_paths[idx]
        try:
            image_np = load_sentinel2_image(image_path) # NumPy (C, H, W)
            image_np = normalize_image_per_image(image_np) # Apply per-image normalization HERE
            image_tensor = torch.from_numpy(image_np).float() # Convert to tensor

            if self.transform:
                image_tensor = self.transform(image_tensor) # Apply remaining transforms

            return image_tensor, 0, image_path # Dummy label 0

        except Exception as e:
            print(f"Error loading/processing image {image_path}:")
            traceback.print_exc()
            return None, None, None


# --- Data Transforms (REVISED - NO Normalize Step) ---
train_transforms = transforms.Compose([
    # Input is already a normalized 12-channel Tensor from __getitem__
    # Geometric augmentations work on multi-channel tensors
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=30),
    transforms.Resize((best_image_size, best_image_size), antialias=True),
    # ColorJitter cannot be used easily on 12 channels
])

val_transforms = transforms.Compose([
    # Input is already a normalized 12-channel Tensor from __getitem__
    transforms.Resize((best_image_size, best_image_size), antialias=True), # Resize
    # NO Normalize step here
])


# --- Create Datasets ---
print("Creating and splitting dataset...")
full_dataset_paths_labels = []
class_to_idx_map = {}
class_names = []
idx_counter = 0
# Scan training directory
for class_name in sorted(os.listdir(train_dir)):
     class_dir = os.path.join(train_dir, class_name)
     if os.path.isdir(class_dir) and not class_name.startswith('.'):
         if class_name not in class_to_idx_map:
             class_to_idx_map[class_name] = idx_counter
             class_names.append(class_name)
             idx_counter += 1
         class_idx = class_to_idx_map[class_name]
         for filename in os.listdir(class_dir):
             if filename.lower().endswith(('.tif', '.tiff')):
                 full_dataset_paths_labels.append((os.path.join(class_dir, filename), class_idx))

num_classes = len(class_names)
if num_classes == 0: raise FileNotFoundError(f"No valid class folders found in {train_dir}")
print(f"Found {len(full_dataset_paths_labels)} training images in {num_classes} classes: {class_names}")

# Stratified Split
train_info, val_info = train_test_split(
    full_dataset_paths_labels, train_size=train_ratio, random_state=seed,
    stratify=[label for _, label in full_dataset_paths_labels]
)

# Create Dataset objects using the split lists
train_dataset = Sentinel2Dataset(train_info, transform=train_transforms)
val_tif_dataset = Sentinel2Dataset(val_info, transform=val_transforms) # Use val_transforms for internal val

# Dataset for final predictions
final_validation_dataset = NpyPredictionDataset(validation_dir, transform=val_transforms) # Use val_transforms

# --- Create DataLoaders ---
def collate_fn(batch):
    """ Filter out None samples before creating batch """
    batch = list(filter(lambda x: x is not None and x[0] is not None, batch))
    if not batch: return None, None, None
    try:
        images = torch.stack([item[0] for item in batch])
        labels = torch.stack([item[1] for item in batch])
        paths = [item[2] for item in batch]
        return images, labels, paths
    except Exception as e:
        print(f"Error in collate_fn: {e}")
        traceback.print_exc()
        return None, None, None

train_loader = DataLoader(train_dataset, batch_size=best_batch_size, shuffle=True, num_workers=num_workers, generator=torch.Generator().manual_seed(seed), pin_memory=True, collate_fn=collate_fn)
val_loader_split = DataLoader(val_tif_dataset, batch_size=best_batch_size, pin_memory=True, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
final_pred_loader = DataLoader(final_validation_dataset, batch_size=best_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)
print("DataLoaders created.")


# --- Model Definition (ConvNeXt adapted for 12->3 channels internally) ---
class ConvNeXtSentinel2(nn.Module):
    """ ConvNeXt model adapted for 12-channel input using an initial 1x1 conv. """
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        self.channel_projector = nn.Conv2d(12, 3, kernel_size=1, bias=False)
        self.convnext_base = models.convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT if pretrained else None)
        if pretrained:
            # print("Initializing 1x1 channel projector weights...") # Less verbose
            with torch.no_grad():
                try: # Init using averaged ResNet weights
                    rn_weights = ResNet50_Weights.IMAGENET1K_V1.get_state_dict()['conv1.weight']
                    rn_weights_avg = rn_weights.mean(dim=[2, 3])
                    proj_weights = self.channel_projector.weight.data
                    proj_weights.zero_()
                    for out_ch in range(3):
                        for in_ch in range(12):
                            proj_weights[out_ch, in_ch, 0, 0] = rn_weights_avg[out_ch % 64, in_ch % 3]
                    # print("Initialized conv1x1 using averaged ResNet weights.")
                except Exception as init_e:
                    print(f"Warning: Failed init conv1x1 from ResNet ({init_e}). Using Kaiming.")
                    nn.init.kaiming_normal_(self.channel_projector.weight, mode='fan_out', nonlinearity='relu')
        num_features_in = self.convnext_base.classifier[-1].in_features
        self.convnext_base.classifier = nn.Sequential(
             self.convnext_base.classifier[0], # Keep LayerNorm
             self.convnext_base.classifier[1], # Keep Flatten
             nn.Dropout(p=0.5),              # Add dropout
             nn.Linear(num_features_in, num_classes) # New Linear layer
        )
        nn.init.normal_(self.convnext_base.classifier[-1].weight, 0, 0.01)
        nn.init.zeros_(self.convnext_base.classifier[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_projector(x) # 12 -> 3 channels
        x = self.convnext_base(x) # Pass 3ch tensor to standard ConvNeXt
        return x

# --- Helper Function for Training/Validation Epoch ---
def run_epoch(model, loader, criterion, optimizer, scaler, device, is_training, epoch_num, num_epochs_total, warmup_scheduler=None, current_stage_epoch=0, total_warmup_epochs=0):
    """Runs a single epoch of training or validation."""
    if is_training: model.train()
    else: model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    start_time = time.time()
    context = torch.no_grad() if not is_training else torch.enable_grad()
    loader_desc = "Training" if is_training else "Validation"
    with context:
        for batch_idx, batch_data in enumerate(loader):
            if batch_data is None or batch_data[0] is None: continue
            inputs, targets, _ = batch_data # Unpack
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.amp.autocast(device_type=device.type, enabled=USE_CUDA):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            if is_training:
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"WARNING: NaN/Inf loss at E{epoch_num} B{batch_idx+1}. Skip.")
                    optimizer.zero_grad(set_to_none=True); continue
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                # Optional Clipping
                # scaler.unscale_(optimizer)
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                if warmup_scheduler and epoch_num <= total_warmup_epochs: warmup_scheduler.step()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += targets.size(0)
            correct_predictions += (predicted == targets).sum().item()
            if is_training and (batch_idx + 1) % 100 == 0: print(f'  E{epoch_num} Step [{batch_idx + 1}/{len(loader)}], Loss: {loss.item():.4f}')
    epoch_duration = time.time() - start_time
    avg_loss = running_loss / total_samples if total_samples > 0 else 0
    accuracy = 100. * correct_predictions / total_samples if total_samples > 0 else 0
    print(f'Epoch [{epoch_num}/{num_epochs_total}] FINISHED ({loader_desc}) - Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%, Time: {epoch_duration:.2f}s')
    return avg_loss, accuracy


# --- Main Execution Logic ---
if __name__ == '__main__':

    # --- Create Model ---
    model = ConvNeXtSentinel2(num_classes=num_classes, pretrained=True) # Use the specific class
    model.to(DEVICE)

    # --- Loss Function ---
    criterion = nn.CrossEntropyLoss() # Define criterion here

    # --- Staged Fine-tuning Variables ---
    overall_best_val_loss = float('inf')
    best_model_state_dict = None
    total_epochs_run = 0
    scaler = torch.amp.GradScaler(device=DEVICE.type, enabled=USE_CUDA) # Correct syntax

    # --- Stage 1 ---
    print("\n--- Stage 1: Training Projector and Classifier Head ---")
    stage_epochs = stage1_epochs
    stage_lr = lr_stage1
    # Configure requires_grad for Stage 1
    for name, param in model.named_parameters():
        if name.startswith('channel_projector.') or name.startswith('convnext_base.classifier.'):
            param.requires_grad = True
        else:
            param.requires_grad = False
    print(f"Trainable parameters for Stage 1: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=stage_lr,
                            weight_decay=best_weight_decay)
    if warmup_epochs > 0:
        warmup_steps_per_epoch_st1 = len(train_loader);
        total_warmup_steps_st1 = warmup_steps_per_epoch_st1 * warmup_epochs
        lr_lambda = lambda cs: initial_warmup_lr / stage_lr * (
                    1.0 - float(cs + 1) / float(max(1, total_warmup_steps_st1))) + float(cs + 1) / float(
            max(1, total_warmup_steps_st1)) if cs < total_warmup_steps_st1 else 1.0
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda);
        print(f"Using linear warmup for {warmup_epochs} epochs in Stage 1.")
    else:
        warmup_scheduler = None
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience // 2,
                                                           verbose=False)
    epochs_without_improvement = 0

    for epoch in range(stage_epochs):
        epoch_num = total_epochs_run + 1
        run_epoch(model, train_loader, criterion, optimizer, scaler, DEVICE, True, epoch_num, num_epochs,
                  warmup_scheduler, epoch, warmup_epochs)
        avg_val_loss, val_accuracy = run_epoch(model, val_loader_split, criterion, None, scaler, DEVICE, False,
                                               epoch_num, num_epochs)
        if epoch >= warmup_epochs: scheduler.step(avg_val_loss)
        print(f'End of Epoch {epoch_num} - Current LR: {optimizer.param_groups[0]["lr"]:.6e}')
        if avg_val_loss < overall_best_val_loss:
            overall_best_val_loss = avg_val_loss;
            epochs_without_improvement = 0
            best_model_state_dict = model.state_dict();
            torch.save(best_model_state_dict, model_save_path)
            print(f'---> Overall Validation Loss Improved to {overall_best_val_loss:.4f}, model saved.')
        else:
            epochs_without_improvement += 1;
            print(f'---> Stage 1 Val loss did not improve for {epochs_without_improvement} epochs.')
            if epochs_without_improvement >= patience: print(
                f'Early stopping triggered during Stage 1 at epoch {epoch_num}.'); break
        total_epochs_run = epoch_num

    # --- Stage 2 ---
    print("\n--- Stage 2: Unfreezing Last Backbone Stage ---")
    if best_model_state_dict:
        model.load_state_dict(best_model_state_dict)
    else:
        print("Warning: No best model from Stage 1, continuing with last state.")
    unfrozen_stage2 = False;
    last_stage_idx = -1
    if hasattr(model, 'convnext_base') and hasattr(model.convnext_base, 'features') and isinstance(
            model.convnext_base.features, nn.Sequential):
        for i in range(len(model.convnext_base.features) - 1, -1, -1):
            if isinstance(model.convnext_base.features[i],
                          (nn.Sequential, models.convnext.CNBlock)): last_stage_idx = i; break
        if last_stage_idx != -1:
            for param in model.convnext_base.features[last_stage_idx].parameters(): param.requires_grad = True
            print(f"Unfroze last backbone stage (index {last_stage_idx}) for Stage 2.")
            unfrozen_stage2 = True
        else:
            print("Warning: Could not find last stage to unfreeze.")

    if unfrozen_stage2:
        print(f"Trainable parameters for Stage 2: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_stage2,
                                weight_decay=best_weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience // 2,
                                                         verbose=False)
        epochs_without_improvement = 0
        for epoch in range(stage2_epochs):
            epoch_num = total_epochs_run + 1
            run_epoch(model, train_loader, criterion, optimizer, scaler, DEVICE, True, epoch_num,
                      num_epochs)  # No warmup
            avg_val_loss, val_accuracy = run_epoch(model, val_loader_split, criterion, None, scaler, DEVICE, False,
                                                   epoch_num, num_epochs)
            scheduler.step(avg_val_loss)
            print(f'End of Epoch {epoch_num} - Current LR: {optimizer.param_groups[0]["lr"]:.6e}')
            if avg_val_loss < overall_best_val_loss:
                overall_best_val_loss = avg_val_loss;
                epochs_without_improvement = 0
                best_model_state_dict = model.state_dict();
                torch.save(best_model_state_dict, model_save_path)
                print(f'---> Overall Validation Loss Improved to {overall_best_val_loss:.4f}, model saved.')
            else:
                epochs_without_improvement += 1;
                print(f'---> Stage 2 Val loss did not improve for {epochs_without_improvement} epochs.')
                if epochs_without_improvement >= patience: print(
                    f'Early stopping triggered during Stage 2 at epoch {epoch_num}.'); break
            total_epochs_run = epoch_num
    else:
        print("Skipping Stage 2 training.")

    # --- Stage 3 ---
    print("\n--- Stage 3: Unfreezing Second-to-Last Backbone Stage ---")
    if best_model_state_dict:
        model.load_state_dict(best_model_state_dict)
    else:
        print("Warning: No best model state from previous stages, continuing.")
    unfrozen_stage3 = False;
    second_last_stage_idx = -1
    if hasattr(model, 'convnext_base') and hasattr(model.convnext_base, 'features') and isinstance(
            model.convnext_base.features, nn.Sequential):
        stages_found = 0
        for i in range(len(model.convnext_base.features) - 1, -1, -1):
            if isinstance(model.convnext_base.features[i], (nn.Sequential, models.convnext.CNBlock)):
                stages_found += 1
                if stages_found == 2: second_last_stage_idx = i; break
        if second_last_stage_idx != -1:
            for param in model.convnext_base.features[second_last_stage_idx].parameters(): param.requires_grad = True
            print(f"Unfroze second-to-last backbone stage (index {second_last_stage_idx}) for Stage 3.")
            unfrozen_stage3 = True
        else:
            print("Warning: Could not find second-to-last stage for Stage 3.")

    if unfrozen_stage3:
        print(f"Trainable parameters for Stage 3: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_stage3,
                                weight_decay=best_weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience // 2,
                                                         verbose=False)
        epochs_without_improvement = 0
        for epoch in range(stage3_epochs):
            epoch_num = total_epochs_run + 1
            run_epoch(model, train_loader, criterion, optimizer, scaler, DEVICE, True, epoch_num, num_epochs)
            avg_val_loss, val_accuracy = run_epoch(model, val_loader_split, criterion, None, scaler, DEVICE, False,
                                                   epoch_num, num_epochs)
            scheduler.step(avg_val_loss)
            print(f'End of Epoch {epoch_num} - Current LR: {optimizer.param_groups[0]["lr"]:.6e}')
            if avg_val_loss < overall_best_val_loss:
                overall_best_val_loss = avg_val_loss;
                epochs_without_improvement = 0
                best_model_state_dict = model.state_dict();
                torch.save(best_model_state_dict, model_save_path)
                print(f'---> Overall Validation Loss Improved to {overall_best_val_loss:.4f}, model saved.')
            else:
                epochs_without_improvement += 1;
                print(f'---> Stage 3 Val loss did not improve for {epochs_without_improvement} epochs.')
                if epochs_without_improvement >= patience: print(
                    f'Early stopping triggered during Stage 3 at epoch {epoch_num}.'); break
            total_epochs_run = epoch_num
    else:
        print("Skipping Stage 3 training.")

    # --- Final Report ---
    if best_model_state_dict is not None:
        print(f'\nBest model saved to {model_save_path} with validation loss: {overall_best_val_loss:.4f}')
    else:
        print("\nTraining completed, but no improvement observed over initial state (or validation failed).")

    print("Script finished.")
