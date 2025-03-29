import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import models, transforms # Keep torchvision transforms
import rasterio
import numpy as np
import pandas as pd
import os
import random
from torch.cuda.amp import GradScaler, autocast
from torchvision.models import ResNet50_Weights, ConvNeXt_Base_Weights
from sklearn.model_selection import train_test_split
import traceback

# --- Setup ---
if not os.path.exists('./outputs'):
    os.mkdir('./outputs')

# Define the paths
train_dir = './ds/images/remote_sensing/otherDatasets/sentinel_2/tif'
validation_dir = './testset/testset'
model_save_path = 'convnext_final_model.pth'
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
best_learning_rate = 1e-5
best_weight_decay =  3.7553e-05
best_optimizer_name = 'AdamW'

num_epochs = 100
warmup_epochs = 5
initial_warmup_lr = 1e-6
num_workers = 4
train_ratio = 0.9
patience = 10
num_classes = 10

# --- Normalization Stats (Defined as Tensors for transforms.Normalize) ---
print("Using predefined 12-channel normalization stats.")
train_mean_12ch = torch.tensor([1353.73, 1117.20, 1041.88, 946.55, 1199.19, 2003.01, 2374.01, 2301.22, 732.18, 1820.70, 1118.20, 2599.78], dtype=torch.float32)
train_std_12ch = torch.tensor([245.21, 327.28, 388.61, 586.99, 565.84, 859.51, 1085.12, 1108.11, 403.78, 1001.43, 759.43, 1229.73], dtype=torch.float32)
# Add epsilon for safety BEFORE passing to Normalize
train_std_12ch = torch.clamp(train_std_12ch, min=1e-7)


# --- Data Loading Function ---
def load_sentinel2_image(filepath):
    """Loads a Sentinel-2 image (12 bands), returns NumPy CHW."""
    if filepath.endswith('.tif'):
        with rasterio.open(filepath) as src:
            if src.count < 13: raise ValueError(f"Expected >=13 bands, got {src.count} in {filepath}")
            bands = list(range(1, 10)) + list(range(11, 14))
            image = src.read(bands) # Reads as (C, H, W)
    elif filepath.endswith('.npy'):
        image = np.load(filepath)
        if image.shape[0] == 13:
            image = np.concatenate((image[:9], image[10:]), axis=0) # (C, H, W)
        # Ensure CHW format if loaded as HWC
        if len(image.shape) == 3 and image.shape[0] != 12 and image.shape[2] == 12:
            image = image.transpose(2, 0, 1) # HWC -> CHW
        elif image.shape[0] != 12:
             raise ValueError(f"Unexpected shape for .npy {filepath}: {image.shape}")
    else:
        raise ValueError("Unsupported file type.")
    # NO normalization here, done in transforms now
    return image.astype(np.float32) # Ensure float32


# --- Dataset Class (Simplified __getitem__) ---
class Sentinel2Dataset(Dataset):
    """Custom Dataset for Sentinel-2 images. Returns Tensor CHW (12 channels)."""
    def __init__(self, paths_labels, transform=None): # Takes paths_labels directly
        self.paths_labels = paths_labels
        self.transform = transform
        # Get classes/mapping if this is the training dataset
        if isinstance(paths_labels[0][1], int): # Check if labels are indices
            unique_labels = sorted(list(set(l for _, l in paths_labels)))
            self.classes = [f"class_{i}" for i in unique_labels] # Placeholder names
            self.class_to_idx = {name: i for i, name in enumerate(self.classes)}


    def __len__(self):
        return len(self.paths_labels)

    def __getitem__(self, idx):
        image_path, label = self.paths_labels[idx]
        try:
            image_np = load_sentinel2_image(image_path) # NumPy (C, H, W)

            # Convert to Tensor FIRST (Tensor CHW)
            image_tensor = torch.from_numpy(image_np).float()

            label_tensor = torch.tensor(label, dtype=torch.long)

            # Apply transforms (which now include Normalize)
            if self.transform:
                image_tensor = self.transform(image_tensor) # Pass the TENSOR

            return image_tensor, label_tensor, image_path

        except Exception as e:
            print(f"Error loading/processing image {image_path}:")
            traceback.print_exc()
            return None, None, None # Signal error

# --- Data Transforms (REVISED AGAIN - No ToTensor, Normalize uses 12ch stats) ---
train_transforms = transforms.Compose([
    # Input is already a 12-channel Tensor from __getitem__
    # Geometric augmentations work on multi-channel tensors
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=30),
    transforms.Resize((best_image_size, best_image_size), antialias=True), # Apply Resize
    # Cannot use ColorJitter, RandomResizedCrop easily on 12 channels here
    transforms.Normalize(mean=train_mean_12ch, std=train_std_12ch) # Use 12-channel stats
])

val_transforms = transforms.Compose([
    # Input is already a 12-channel Tensor from __getitem__
    transforms.Resize((best_image_size, best_image_size), antialias=True), # Resize
    transforms.Normalize(mean=train_mean_12ch, std=train_std_12ch) # Use 12-channel stats
])

# --- Create Datasets ---
print("Creating and splitting dataset...")
full_dataset_paths_labels = [] # Store (path, label_idx)
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

num_classes = len(class_names) # Update num_classes
print(f"Found {len(full_dataset_paths_labels)} training images in {num_classes} classes.")
print(f"Class names: {class_names}")


# Stratified Split
train_info, val_info = train_test_split(
    full_dataset_paths_labels,
    train_size=train_ratio,
    random_state=seed,
    stratify=[label for _, label in full_dataset_paths_labels]
)

# Create Dataset objects using the split lists
train_dataset = Sentinel2Dataset(train_info, transform=train_transforms)
train_dataset.classes = class_names # Manually assign class info if needed later
train_dataset.class_to_idx = class_to_idx_map

val_tif_dataset = Sentinel2Dataset(val_info, transform=val_transforms)
val_tif_dataset.classes = class_names
val_tif_dataset.class_to_idx = class_to_idx_map


# Validation dataset for FINAL evaluation (using .npy files)
class NpyPredictionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_paths = sorted([os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.npy')])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image_path = self.file_paths[idx]
        try:
            image_np = load_sentinel2_image(image_path) # NumPy (C, H, W)
            # NO normalization on NumPy here

            image_tensor = torch.from_numpy(image_np).float() # Tensor (C, H, W)

            if self.transform:
                image_tensor = self.transform(image_tensor) # Apply val_transforms

            return image_tensor, 0, image_path # Dummy label 0

        except Exception as e:
            print(f"Error loading/processing image {image_path}:")
            traceback.print_exc()
            return None, None, None

final_validation_dataset = NpyPredictionDataset(validation_dir, transform=val_transforms)


# --- Create DataLoaders ---
def collate_fn(batch):
    """ Filter out None samples before creating batch """
    batch = list(filter(lambda x: x is not None and x[0] is not None, batch))
    if not batch: return None, None, None
    images = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    paths = [item[2] for item in batch]
    return images, labels, paths

train_loader = DataLoader(train_dataset, batch_size=best_batch_size, shuffle=True, num_workers=num_workers,
                          generator=torch.Generator().manual_seed(seed), pin_memory=True, collate_fn=collate_fn)
val_loader_split = DataLoader(val_tif_dataset, batch_size=best_batch_size, pin_memory=True, shuffle=False,
                              num_workers=num_workers, collate_fn=collate_fn)
final_pred_loader = DataLoader(final_validation_dataset, batch_size=best_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)

print("DataLoaders created.")


# --- Model Definition (ConvNeXt adapted for 12->3 channels internally) ---
class ConvNeXtSentinel2(nn.Module):
    """ ConvNeXt model adapted for 12-channel input using an initial 1x1 conv. """
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        # --- 1. Define the 1x1 Channel Projector ---
        self.channel_projector = nn.Conv2d(12, 3, kernel_size=1, bias=False)

        # --- 2. Load base ConvNeXt model ---
        self.convnext_base = models.convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT if pretrained else None)

        # --- 3. Initialize Projector Weights ---
        if pretrained:
            print("Initializing 1x1 channel projector weights...")
            with torch.no_grad():
                # Use spatially averaged weights from ResNet50 conv1 as a starting point
                try:
                    rn_weights = ResNet50_Weights.IMAGENET1K_V1.get_state_dict()['conv1.weight']
                    rn_weights_avg = rn_weights.mean(dim=[2, 3]) # Avg spatial dims -> [64, 3]
                    proj_weights = self.channel_projector.weight.data # [3, 12, 1, 1]
                    proj_weights.zero_()
                    for out_ch in range(3):
                        for in_ch in range(12):
                            source_out_ch = out_ch
                            source_in_ch = in_ch % 3
                            proj_weights[out_ch, in_ch, 0, 0] = rn_weights_avg[source_out_ch, source_in_ch]
                    print("Initialized conv1x1 using averaged ResNet weights.")
                except Exception as init_e:
                    print(f"Warning: Failed to initialize conv1x1 from ResNet ({init_e}). Using Kaiming init.")
                    nn.init.kaiming_normal_(self.channel_projector.weight, mode='fan_out', nonlinearity='relu')

        # --- 4. Store original components (if needed, but likely not) ---
        # self.features = self.convnext_base.features # Direct access might change
        # self.avgpool = self.convnext_base.avgpool
        # original_classifier = self.convnext_base.classifier

        # --- 5. Replace the classifier ---
        num_features_in = self.convnext_base.classifier[-1].in_features
        self.convnext_base.classifier = nn.Sequential(
             # Keep structure similar if needed (LayerNorm, Flatten)
             self.convnext_base.classifier[0], # LayerNorm
             self.convnext_base.classifier[1], # Flatten
             nn.Dropout(p=0.5),
             nn.Linear(num_features_in, num_classes)
        )
        print(f"Replaced classifier head for {num_classes} classes.")

        # --- 6. Initialize New Classifier Head ---
        nn.init.normal_(self.convnext_base.classifier[-1].weight, 0, 0.01)
        nn.init.zeros_(self.convnext_base.classifier[-1].bias)

        # --- 7. Set requires_grad for Fine-Tuning ---
        print("Configuring model for fine-tuning...")
        # Freeze everything initially
        for param in self.parameters():
            param.requires_grad = False
        # Unfreeze the projector, pre-classifier norm (if exists), and classifier
        for param in self.channel_projector.parameters(): # Unfreeze projector
            param.requires_grad = True
        for param in self.convnext_base.classifier.parameters(): # Unfreeze new classifier
            param.requires_grad = True

        # Unfreeze the last stage of the backbone
        if hasattr(self.convnext_base, 'features') and isinstance(self.convnext_base.features, nn.Sequential):
            last_stage_idx = -1
            for i in range(len(self.convnext_base.features) - 1, -1, -1):
                 if isinstance(self.convnext_base.features[i], (nn.Sequential, models.convnext.CNBlock)):
                     last_stage_idx = i
                     break
            if last_stage_idx != -1:
                for param in self.convnext_base.features[last_stage_idx].parameters():
                    param.requires_grad = True
                print(f"Unfroze channel_projector, classifier, and last stage (idx {last_stage_idx}).")
            else:
                 print("Unfroze channel_projector and classifier only.")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Project 12 channels to 3
        x = self.channel_projector(x)
        # 2. Pass through the *original* ConvNeXt base model's forward path
        x = self.convnext_base(x) # This will use the modified classifier internally
        return x


model = ConvNeXtSentinel2(num_classes=num_classes, pretrained=True) # Use the specific class
model.to(DEVICE)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {trainable_params}")


# --- Loss Function, Optimizer, and Scheduler ---
criterion = nn.CrossEntropyLoss()
optimizer_params = filter(lambda p: p.requires_grad, model.parameters())
if best_optimizer_name.lower() == 'adamw':
    optimizer = torch.optim.AdamW(optimizer_params, lr=best_learning_rate, weight_decay=best_weight_decay)
elif best_optimizer_name.lower() == 'adam':
    optimizer = torch.optim.Adam(optimizer_params, lr=best_learning_rate, weight_decay=best_weight_decay)
else:
    raise ValueError(f"Invalid optimizer name: {best_optimizer_name}")

# Warmup Scheduler
if warmup_epochs > 0:
    warmup_steps_per_epoch = len(train_loader)
    total_warmup_steps = warmup_steps_per_epoch * warmup_epochs
    def lr_lambda(current_step):
        if current_step < total_warmup_steps:
            scale_factor = float(current_step + 1) / float(max(1, total_warmup_steps))
            lr_mult = initial_warmup_lr / best_learning_rate * (1.0 - scale_factor) + scale_factor
            return lr_mult
        return 1.0
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    print(f"Using linear warmup for {warmup_epochs} epochs ({total_warmup_steps} steps).")
else:
    warmup_scheduler = None

# Main epoch-level scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience // 2)

scaler = torch.amp.GradScaler(device=DEVICE.type, enabled=USE_CUDA) # Correct syntax

# --- Training Loop ---
best_val_loss = float('inf')
epochs_without_improvement = 0
best_model_state = None

print("\n--- Starting Final Training ---")
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    correct_train_predictions = 0
    total_train_samples = 0
    print(f'\nEpoch {epoch + 1}/{num_epochs}')

    for batch_idx, batch_data in enumerate(train_loader):
        if batch_data is None or batch_data[0] is None:
            # print(f"Warning: Skipping empty/invalid batch {batch_idx}") # Less verbose
            continue
        inputs, targets, _ = batch_data # Unpack correctly

        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)

        optimizer.zero_grad(set_to_none=True) # Use set_to_none=True

        with torch.amp.autocast(device_type=DEVICE.type, enabled=USE_CUDA):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        # Check for NaN loss before scaling/backward
        if torch.isnan(loss):
            print(f"WARNING: NaN loss detected at Epoch {epoch+1}, Batch {batch_idx+1}. Skipping batch.")
            optimizer.zero_grad() # Clear potentially bad gradients
            continue # Skip this batch

        scaler.scale(loss).backward()

        # Optional gradient clipping
        # scaler.unscale_(optimizer)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        # Step iteration-level scheduler (warmup phase)
        if warmup_scheduler and epoch < warmup_epochs:
             warmup_scheduler.step()

        # --- Accumulate ---
        total_train_loss += loss.item() * inputs.size(0) # Use inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_train_samples += targets.size(0)
        correct_train_predictions += (predicted == targets).sum().item()

        if (batch_idx + 1) % 50 == 0: # Print less frequently
            print(f'  Train Step [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    avg_train_loss = total_train_loss / total_train_samples if total_train_samples > 0 else 0
    train_accuracy = 100. * correct_train_predictions / total_train_samples if total_train_samples > 0 else 0
    print(f'Epoch [{epoch + 1}/{num_epochs}] Avg Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')

    # --- Validation (using val_loader_split with .tif files) ---
    model.eval()
    total_val_loss = 0
    correct_val_predictions = 0
    total_val_samples = 0
    with torch.no_grad():
        for batch_data in val_loader_split: # Use the correct loader for internal val
            if batch_data is None or batch_data[0] is None:
                # print(f"Warning: Skipping empty/invalid validation batch") # Less verbose
                continue
            inputs, targets, _ = batch_data # Unpack

            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            with torch.amp.autocast(device_type=DEVICE.type, enabled=USE_CUDA): # Add autocast
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            # Check for NaN loss in validation (less likely but possible)
            if torch.isnan(loss):
                 print(f"WARNING: NaN loss detected during validation epoch {epoch+1}.")
                 continue

            total_val_loss += loss.item() * inputs.size(0) # Use inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_val_samples += targets.size(0)
            correct_val_predictions += (predicted == targets).sum().item()

    # Check if validation happened at all
    if total_val_samples == 0:
        print("Warning: No valid validation samples processed in this epoch.")
        avg_val_loss = float('inf') # Assign high loss if no validation
        val_accuracy = 0.0
    else:
        avg_val_loss = total_val_loss / total_val_samples
        val_accuracy = 100. * correct_val_predictions / total_val_samples

    print(f'Epoch [{epoch + 1}/{num_epochs}] Avg Validation Loss: {avg_val_loss:.4f}, Validation Acc: {val_accuracy:.2f}%')

    # Step the EPOCH level scheduler AFTER the warmup phase
    if epoch >= warmup_epochs:
         scheduler.step(avg_val_loss) # Use validation loss

    current_lr = optimizer.param_groups[0]['lr'] # Get current LR
    print(f'Current LR: {current_lr:.6e}')

    # Early Stopping Logic
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_without_improvement = 0
        best_model_state = model.state_dict()
        torch.save(best_model_state, model_save_path) # Save the improved model
        print(f'---> Validation Loss Improved to {best_val_loss:.4f}, model saved.')
    else:
        epochs_without_improvement += 1
        print(f'---> Validation loss did not improve for {epochs_without_improvement} epochs.')
        if epochs_without_improvement >= patience:
            print(f'Early stopping triggered at epoch {epoch + 1}.')
            break

# --- After the loop ---
if best_model_state is not None:
    print(f'Best model saved during training to {model_save_path} with validation loss: {best_val_loss:.4f}')
    # Load the best model state back into the model for final prediction step
    model.load_state_dict(best_model_state)
else:
    print("Training finished without improvement or saving a model.")
    # Keep the last state if no improvement was ever made
    best_model = model # 'model' already holds the last state

print("Final training finished!")


# --- Final Prediction on Validation/Test Set (.npy files) and Create CSV ---
print("\n--- Generating predictions for submission ---")
# Ensure the best model weights are loaded
best_model_to_pred = ConvNeXtSentinel2(num_classes=num_classes, pretrained=False) # Recreate architecture
try:
    # Use weights_only=True for security when loading external/saved files
    best_model_to_pred.load_state_dict(torch.load(model_save_path, map_location=DEVICE, weights_only=True))
    print(f"Loaded best model weights from {model_save_path}")
except FileNotFoundError:
    print(f"Error: Model file not found at {model_save_path}. Using last model state if available.")
    if best_model_state is not None:
         best_model_to_pred.load_state_dict(best_model_state) # Use last saved best state
    else:
         print("WARNING: No best model state found, predicting with potentially untrained model!")
         best_model_to_pred = model # Use the model from end of training

best_model_to_pred.to(DEVICE)
best_model_to_pred.eval() # Set to evaluation mode

all_predictions = []
all_filenames = []

with torch.no_grad():
    for batch_data in final_pred_loader: # Use the loader with .npy files
        if batch_data is None or batch_data[0] is None:
            print(f"Warning: Skipping empty/invalid prediction batch")
            continue
        inputs, _, image_paths = batch_data # Get filenames!

        inputs = inputs.to(DEVICE)
        with torch.amp.autocast(device_type=DEVICE.type, enabled=USE_CUDA):
            outputs = best_model_to_pred(inputs) # Use the loaded best model
        _, predicted = torch.max(outputs, 1)

        # Use class names defined earlier based on training data folders
        predicted_classes = [class_names[pred.item()] for pred in predicted]
        all_predictions.extend(predicted_classes)
        all_filenames.extend([os.path.basename(path) for path in image_paths])


# Create the DataFrame with 'test_id' and 'label'
predictions_df = pd.DataFrame({
    'test_id': [os.path.splitext(filename)[0] for filename in all_filenames],
    'label': all_predictions
})

predictions_csv_path = "_track2.csv" # Ensure correct submission filename
predictions_df.to_csv(predictions_csv_path, index=False)
print(f"Submission predictions saved to: {predictions_csv_path}")

