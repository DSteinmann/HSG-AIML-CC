# evaluate_resnet50_sentinel2.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import pandas as pd
from torchgeo.models import resnet50, ResNet152_Weights, ResNet50_Weights
from torchvision import models
from torchvision.models import resnext101_32x8d, convnext_large, ConvNeXt_Large_Weights, convnext_base, \
    ConvNeXt_Base_Weights, resnet101, ResNet101_Weights
from torchvision.transforms import transforms

train_mean = [1353.727294921875,
              1117.2015380859375,
              1041.88427734375,
              946.5543212890625,
              1199.1885986328125,
              2003.0074462890625,
              2374.008056640625,
              2301.218994140625,
              732.1814575195312,
              1820.696044921875,
              1118.202392578125,
              2599.78369140625]
train_std = [245.2095947265625,
             327.2839050292969,
             388.6127624511719,
             586.9910888671875,
             565.8442993164062,
             859.5098876953125,
             1085.123291015625,
             1108.106689453125,
             403.7764587402344,
             1001.4296875,
             759.4329833984375,
             1229.7283935546875]


class ResNet50Sentinel2(nn.Module):
    # ... (same as before)
    def __init__(self, num_classes, pretrained=False):
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

# --- Test Dataset for .npy files ---
class TestDatasetNPY(Dataset):
    def __init__(self, root_dir, label_map=None, transform=None):
        """
        Dataset for loading test data from .npy files.

        Args:
            root_dir (str): Path to the directory containing the .npy files.
            label_map (dict, optional): Mapping from numerical labels to string labels.
                                         Defaults to None.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.file_paths = sorted([os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(".npy")])
        self.label_map = label_map

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        image = np.load(file_path)  # Load the .npy file
        # Transpose the dimensions to (Channels, Height, Width)
        image = image.transpose((2, 0, 1))
        image_tensor = torch.from_numpy(image).float() # Ensure it's a float tensor

        if self.transform:
            image_tensor = self.transform(image_tensor)

        base_name = os.path.basename(file_path)
        test_id_str = base_name.replace(".npy", "").replace("test_", "")
        test_id = int(test_id_str)

        return image_tensor, test_id


def main():
    # --- Configuration ---
    model_path = "convnext_final_model.pth"  # Path to your trained model file
    test_data_dir = "./testset/testset"  # Path to the directory containing .npy test files
    output_csv_path = "_track2.csv"
    num_classes = 10  # Replace with the number of classes your model was trained on

    # --- Define Label Mapping (if available from training) ---
    # You might need to load this from a saved file or define it here
    label_map = {0: 'AnnualCrop', 1: 'Forest', 2: 'HerbaceousVegetation', 3: 'Highway', 4: 'Industrial', 5: 'Pasture', 6: 'PermanentCrop', 7: 'Residential', 8: 'River', 9: 'SeaLake'} # Example target labels

    # If you saved the label_map during training, you can load it:
    # import json
    # with open("label_map.json", "r") as f:
    #     label_map = json.load(f)
    #
    # For the desired output format, you'll need the specific string labels
    if label_map:
        reverse_label_map = {v: k for k, v in label_map.items()}
        target_label_map = {0: 'AnnualCrop', 1: 'Forest', 2: 'HerbaceousVegetation', 3: 'Highway', 4: 'Industrial', 5: 'Pasture', 6: 'PermanentCrop', 7: 'Residential', 8: 'River', 9: 'SeaLake'} # Example target labels
        # Adjust this based on your actual class names and mapping

    # --- Create Test Dataset ---
    test_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Normalize(mean=train_mean, std=train_std)
    ])

    test_dataset = TestDatasetNPY(test_data_dir, label_map=label_map, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False) # No need to shuffle for evaluation

    # --- Load the Model ---
    model = ConvNeXtSentinel2(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(model_path))
    model.eval() # Set the model to evaluation mode

    # --- Determine Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    # --- Make Predictions ---
    predictions = []
    with torch.no_grad():
        for inputs, test_ids in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1) # Get the index of the max log-probability
            predicted_labels = predicted.cpu().numpy()
            test_ids_numpy = test_ids.cpu().numpy()
            for i in range(len(test_ids_numpy)):
                test_id = test_ids_numpy[i]
                label_index = predicted_labels[i]
                predicted_label_str = None
                if label_map and label_index in label_map:
                    # If you have a simple class name (like Class_0) in label_map
                    predicted_label_str = label_map[label_index]
                    # You might need to map this to the final desired string label
                    if target_label_map and label_index in target_label_map:
                        predicted_label_str = target_label_map[label_index]
                else:
                    predicted_label_str = f"Class_{label_index}" # Fallback if no mapping

                predictions.append({'test_id': test_id, 'label': predicted_label_str})

    # --- Create and Save CSV ---
    df = pd.DataFrame(predictions)
    df.to_csv(output_csv_path, index=False)

    print(f"Predictions saved to {output_csv_path}")



if __name__ == "__main__":

    npy_file_path = "./testset/testset/test_1.npy"  # Replace with the actual path
    data = np.load(npy_file_path)
    print(f"Shape of {npy_file_path}: {data.shape}")
    test_data_dir = "./testset/testset"
    main()