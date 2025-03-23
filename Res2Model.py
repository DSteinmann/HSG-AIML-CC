# runnable_resnet50_sentinel2_optimized_mac.py
import optuna
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
from torchgeo.models import ResNet152_Weights, resnet152, resnet50, ResNet50_Weights
from torchvision.models import convnext_base, ConvNeXt_Large_Weights, resnet101, ResNet101_Weights, \
    ConvNeXt_Base_Weights
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import os
import numpy as np
import rasterio
import random


class ConvNeXtSentinel2(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        self.convnext = convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT if pretrained else None)

        first_conv_layer = self.convnext.features[0][0]
        original_out_channels = first_conv_layer.out_channels
        original_kernel_size = first_conv_layer.kernel_size
        original_stride = first_conv_layer.stride
        original_padding = first_conv_layer.padding
        original_bias = first_conv_layer.bias is not None

        self.convnext.features[0][0] = nn.Conv2d(
            12,
            original_out_channels,
            kernel_size=original_kernel_size,
            stride=original_stride,
            padding=original_padding,
            bias=original_bias
        )

        if pretrained:
            # Initialize weights for the new first layer (12 input channels)
            nn.init.kaiming_normal_(self.convnext.features[0][0].weight, mode='fan_out', nonlinearity='relu')
            if self.convnext.features[0][0].bias is not None:
                nn.init.zeros_(self.convnext.features[0][0].bias)

            # Optionally freeze earlier layers (excluding the classifier)
            for name, param in self.convnext.named_parameters():
                if 'classifier' not in name: # You might need to adjust this based on the exact layer names
                    param.requires_grad = False

        num_features = self.convnext.classifier[-1].in_features
        self.convnext.classifier[-1] = nn.Linear(num_features, num_classes)
        nn.init.normal_(self.convnext.classifier[-1].weight, 0, 0.01)
        nn.init.zeros_(self.convnext.classifier[-1].bias)

    def forward(self, x):
        return self.convnext(x)

# --- Model Definition ResNet ---
class ResNet50Sentinel2(nn.Module):
    # ... (same as before)
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        self.resnet50 = resnet50(weights=ResNet50_Weights.SENTINEL2_ALL_MOCO if pretrained else None)
        original_conv1 = self.resnet50.conv1
        self.resnet50.conv1 = nn.Conv2d(12, original_conv1.out_channels,
                                       kernel_size=original_conv1.kernel_size,
                                       stride=original_conv1.stride,
                                       padding=original_conv1.padding,
                                       bias=False if original_conv1.bias is None else True)
        if pretrained:
            original_weights = original_conv1.weight.data
            # Remove the weights corresponding to the 10th channel (index 9)
            indices_to_keep = [i for i in range(original_weights.shape[1]) if i != 9]
            new_weights = original_weights[:, indices_to_keep, :, :]

            # Create a new first convolutional layer with 12 input channels
            self.resnet50.conv1 = nn.Conv2d(12, original_conv1.out_channels,
                                            kernel_size=original_conv1.kernel_size,
                                            stride=original_conv1.stride,
                                            padding=original_conv1.padding,
                                            bias=False if original_conv1.bias is None else True)

            # Initialize the weights of the new first layer with the modified pre-trained weights
            self.resnet50.conv1.weight.data = new_weights

            # Freeze earlier layers (excluding the classifier)
            for name, param in self.resnet50.named_parameters():
                if name not in ['fc.weight', 'fc.bias']:
                    param.requires_grad = False
        else:
            # If not pretrained, initialize the first conv layer for 12 channels
            self.resnet50.conv1 = nn.Conv2d(12, original_conv1.out_channels,
                                            kernel_size=original_conv1.kernel_size,
                                            stride=original_conv1.stride,
                                            padding=original_conv1.padding,
                                            bias=False if original_conv1.bias is None else True)
            nn.init.kaiming_normal_(self.resnet50.conv1.weight, mode='fan_out', nonlinearity='relu')

        num_ftrs = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet50(x)

# --- Sentinel-2 Dataset using rasterio ---
class Sentinel2Folder(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        A dataset for locally stored Sentinel-2 data using rasterio.

        Args:
            root_dir (str): Path to the root directory containing subfolders for labels.
                              Each label folder should contain TIFF files with 13 bands.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.label_map = {}
        label_counter = 0

        for label_folder in sorted(os.listdir(root_dir)):
            label_path = os.path.join(root_dir, label_folder)
            if os.path.isdir(label_path):
                self.label_map[label_counter] = label_folder
                for filename in os.listdir(label_path):
                    if filename.endswith(".tif") or filename.endswith(".tiff"):
                        self.image_paths.append(os.path.join(label_path, filename))
                        self.labels.append(label_counter)
                label_counter += 1

        self.num_classes = len(self.label_map)
        print(f"Found {len(self.image_paths)} images belonging to {self.num_classes} classes.")
        print(f"Label mapping: {self.label_map}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            with rasterio.open(img_path) as src:
                if src.count < 13:
                    raise ValueError(f"TIFF file {img_path} has {src.count} bands, expected at least 13.")
                # Read all bands
                bands = [src.read(i + 1) for i in range(src.count)]
                # Omit the 10th band (index 9)
                selected_bands = [bands[i] for i in range(len(bands)) if i != 9]
                # Stack the selected bands along the first dimension (channels)
                image = np.stack(selected_bands, axis=0).astype(np.float32)
                image_tensor = torch.from_numpy(image)

                if self.transform:
                    image_tensor = self.transform(image_tensor)

                return image_tensor, label

        except rasterio.RasterioIOError as e:
            print(f"Error reading TIFF file {img_path}: {e}")
            # Handle the error appropriately, e.g., return None or a default value
            return None, label  # Or raise the error

        except ValueError as e:
            print(e)
            return None, label

def check_frozen_layers(model):
    frozen_count = 0
    total_count = 0
    for name, param in model.named_parameters():
        total_count += 1
        if not param.requires_grad:
            print(f"Layer '{name}' is frozen (requires_grad=False)")
            frozen_count += 1
    if frozen_count == 0:
        print("All layers in the model are trainable (requires_grad=True)")
    else:
        print(f"Found {frozen_count} frozen layers out of {total_count} total layers.")

def unfreeze_all_layers(model):
    for param in model.parameters():
        param.requires_grad = True
    print("All layers in the model have been unfrozen (requires_grad=True).")

class WarmupLR(_LRScheduler):
    """
    Warmup learning rate scheduler.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_steps (int): Number of warmup steps.
        initial_lr (float): Initial learning rate at the start of warmup.
        target_lr (float): Target learning rate after warmup.
        last_epoch (int): The index of the last epoch when resuming a training.
            Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    """

    def __init__(self, optimizer, warmup_steps, initial_lr, target_lr, last_epoch=-1, verbose=False):
        self.warmup_steps = warmup_steps
        self.initial_lr = initial_lr
        self.target_lr = target_lr
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            alpha = self.last_epoch / float(self.warmup_steps)
            return [self.initial_lr + (self.target_lr - self.initial_lr) * alpha for base_lr in self.base_lrs]
        else:
            # After warmup, keep the target learning rate
            return self.base_lrs

class TransformedDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, label = self.dataset[idx]
        if self.transform:
            data = self.transform(data)
        return data, label

def objective(trial):
    # --- Configuration (Tunable Hyperparameters) ---
    image_size = trial.suggest_categorical('image_size', [128, 256, 512])
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical('optimizer', ['AdamW', 'Adam'])

    seed = 1337
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    root_directory = "./ds/images/remote_sensing/otherDatasets/sentinel_2/tif"  # Replace with your actual data path
    num_epochs = 15  # Train for a few epochs during optimization
    warmup_epochs = 1
    initial_warmup_lr = 1e-6
    num_workers = 8
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1
    patience = 3

    # --- Create Dataset ---
    full_dataset = Sentinel2Folder(root_directory)
    valid_indices = [i for i in range(len(full_dataset)) if full_dataset[i][0] is not None]
    from torch.utils.data import Subset
    total_size = len(valid_indices)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    train_indices = valid_indices[:train_size]
    val_indices = valid_indices[train_size:train_size + val_size]
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    num_classes = full_dataset.num_classes

    # --- Calculate Mean and Std ---
    dataloader_temp = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False, num_workers=num_workers)
    num_bands = 12
    total_sum = torch.zeros(num_bands)
    total_squared_sum = torch.zeros(num_bands)
    total_pixels = 0
    for data, _ in dataloader_temp:
        if data is not None:
            data = data.float()
            total_sum += torch.sum(data, dim=(0, 2, 3))
            total_squared_sum += torch.sum(data ** 2, dim=(0, 2, 3))
            total_pixels += data.numel() // num_bands
    train_mean = total_sum / total_pixels
    train_std = torch.sqrt((total_squared_sum / total_pixels) - (train_mean ** 2))

    # --- Define Data Transforms ---
    train_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=30),
        transforms.ToTensor(),
        transforms.Normalize(mean=train_mean.tolist(), std=train_std.tolist()),
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=train_mean.tolist(), std=train_std.tolist()),
    ])
    train_dataset.transform = train_transforms
    val_dataset.transform = val_transforms

    # --- Create DataLoaders ---
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # --- Determine Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Create Model and Optimizer ---
    model = ResNet50Sentinel2(num_classes=num_classes, pretrained=True).to(device)
    unfreeze_all_layers(model) # Ensure all layers are trainable for optimization

    if optimizer_name == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Invalid optimizer name: {optimizer_name}")

    criterion = nn.CrossEntropyLoss()
    warmup_steps_per_epoch = len(train_loader)
    total_warmup_steps = warmup_steps_per_epoch * warmup_epochs
    warmup_scheduler = WarmupLR(optimizer, total_warmup_steps, initial_warmup_lr, learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience, verbose=False)

    # --- Training Loop ---
    best_val_accuracy = 0.0
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (data, targets) in enumerate(train_loader):
            if data is None:
                continue
            data = data.to(device)
            targets = targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            warmup_scheduler.step()

        # --- Validation ---
        model.eval()
        correct_val_predictions = 0
        total_val_samples = 0
        with torch.no_grad():
            for data, targets in val_loader:
                if data is None:
                    continue
                data = data.to(device)
                targets = targets.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total_val_samples += targets.size(0)
                correct_val_predictions += (predicted == targets).sum().item()

        val_accuracy = correct_val_predictions / total_val_samples if total_val_samples > 0 else 0
        scheduler.step(0) # Optuna needs a metric to optimize, using 0 here as ReduceLROnPlateau needs a loss

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

        trial.report(val_accuracy, epoch)

        # Handle pruning based on intermediate results
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val_accuracy

if __name__ == '__main__':
    # study = optuna.create_study(direction='maximize')
    # study.optimize(objective, n_trials=10)  # Adjust n_trials as needed
    #
    # print("Number of finished trials: {}".format(len(study.trials)))
    # print("Best trial:")
    # trial = study.best_trial
    # print("  Value: {}".format(trial.value))
    # print("  Params: {}".format(trial.params))
    #
    # # --- Train the best model with the found hyperparameters for more epochs ---
    # best_image_size = trial.params['image_size']
    # best_batch_size = trial.params['batch_size']
    # best_learning_rate = trial.params['learning_rate']
    # best_weight_decay = trial.params['weight_decay']
    # best_optimizer_name = trial.params['optimizer']
    # --- Configuration ---
    seed = 1337
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    root_directory = "./ds/images/remote_sensing/otherDatasets/sentinel_2/tif"  # Replace with your actual data path
    batch_size = 16  # Experiment with this
    image_size = 256
    learning_rate = 0.0001454759712929969
    weight_decay = 1.1104061061403061e-05
    num_epochs = 100
    warmup_epochs = 5  # Number of warmup epochs
    initial_warmup_lr = 1e-6  # Very small initial learning rate
    num_workers = 8  # Adjust this based on your CPU cores
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1
    patience = 5  # Number of epochs to wait for improvement

    # --- Create Dataset ---
    full_dataset = Sentinel2Folder(root_directory)
    # Filter out None samples that might have resulted from reading errors
    full_dataset.image_paths = [path for i, path in enumerate(full_dataset.image_paths) if full_dataset[i][0] is not None]
    full_dataset.labels = [label for i, label in enumerate(full_dataset.labels) if full_dataset[i][0] is not None]
    num_classes = full_dataset.num_classes

    # --- Split Dataset ---
    total_size = len(full_dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)  # For reproducibility
    )

    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    # --- Create DataLoaders ---

    # --- Define Data Transforms ---
    train_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=30),
        #CALCULATE THIS TOMORROW
        #transforms.Normalize(mean=train_mean, std=train_std),
        # Add more transformations as needed
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        # transforms.Normalize(mean=train_mean, std=train_std),
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        # transforms.Normalize(mean=train_mean, std=train_std),
    ])

    train_dataset = TransformedDataset(train_dataset, train_transforms)
    val_dataset = TransformedDataset(val_dataset, val_transforms)
    test_dataset = TransformedDataset(test_dataset, test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    # --- Determine Device ---
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # --- Create Model and Move to Device ---
    model = ConvNeXtSentinel2(num_classes=num_classes, pretrained=True).to(device)

    unfreeze_all_layers(model)
    check_frozen_layers(model) # Verify that all are unfrozen

    # --- Define Loss Function and Optimizer ---
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    warmup_steps_per_epoch = len(train_loader)  # Number of steps in one epoch
    total_warmup_steps = warmup_steps_per_epoch * warmup_epochs

    warmup_scheduler = WarmupLR(optimizer, total_warmup_steps, initial_warmup_lr, learning_rate)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience, verbose=True)

    # --- Training Loop with Validation and Early Stopping ---
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_state = None

    for epoch in range(num_epochs):
        # Training
        model.train()
        total_train_loss = 0
        correct_train_predictions = 0
        total_train_samples = 0
        for batch_idx, (data, targets) in enumerate(train_loader):
            if data is None:
                continue
            data = data.to(device)
            targets = targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            warmup_scheduler.step()
            total_train_loss += loss.item() * data.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train_samples += targets.size(0)
            correct_train_predictions += (predicted == targets).sum().item()

            if (batch_idx + 1) % 10 == 0:
                print(
                    f'Epoch [{epoch + 1}/{num_epochs}], Train Step [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        avg_train_loss = total_train_loss / total_train_samples if total_train_samples > 0 else 0
        train_accuracy = correct_train_predictions / total_train_samples if total_train_samples > 0 else 0
        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Average Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}')

        # Validation
        model.eval()
        total_val_loss = 0
        correct_val_predictions = 0
        total_val_samples = 0
        with torch.no_grad():
            for data, targets in val_loader:
                if data is None:
                    continue
                data = data.to(device)
                targets = targets.to(device)
                outputs = model(data)
                loss = criterion(outputs, targets)

                total_val_loss += loss.item() * data.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val_samples += targets.size(0)
                correct_val_predictions += (predicted == targets).sum().item()

        avg_val_loss = total_val_loss / total_val_samples if total_val_samples > 0 else 0
        val_accuracy = correct_val_predictions / total_val_samples if total_val_samples > 0 else 0
        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Average Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

        # Learning Rate Scheduler Step
        if epoch >= warmup_epochs:
            scheduler.step(avg_val_loss)

        # Early Stopping Check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            best_model_state = model.state_dict()  # Save the best model state
        else:
            epochs_without_improvement += 1
            print(
                f'Epoch [{epoch + 1}/{num_epochs}], Validation loss did not improve for {epochs_without_improvement} epochs.')
            if epochs_without_improvement >= patience:
                print(f'Early stopping triggered at epoch {epoch + 1}.')
                break

    # Load the best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print('Loaded best model state based on validation loss.')

    print("Training finished!")

    # --- Evaluation on Test Set ---
    model.eval()
    total_test_loss = 0
    correct_test_predictions = 0
    total_test_samples = 0
    with torch.no_grad():
        for data, targets in test_loader:
            if data is None:
                continue
            data = data.to(device)
            targets = targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets)

            total_test_loss += loss.item() * data.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_test_samples += targets.size(0)
            correct_test_predictions += (predicted == targets).sum().item()

    avg_test_loss = total_test_loss / total_test_samples if total_test_samples > 0 else 0
    test_accuracy = correct_test_predictions / total_test_samples if total_test_samples > 0 else 0
    print(f'Average Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

    # --- Optional: Save the trained model (best state) ---
    torch.save(model.state_dict(), "convnext_sentinel2_trained_best.pth")