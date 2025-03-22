# evaluate_resnet50_sentinel2.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import pandas as pd
from torchvision.models import resnext101_32x8d, convnext_large, ConvNeXt_Large_Weights, convnext_base, \
    ConvNeXt_Base_Weights


class ResNet50Sentinel2(nn.Module):
    def __init__(self, num_classes, pretrained=False):  # Pretrained should be False for loading
        super().__init__()
        self.resnet50 = resnext101_32x8d(weights=None) # Load without pretrained weights initially
        original_conv1 = self.resnet50.conv1
        self.resnet50.conv1 = nn.Conv2d(12, original_conv1.out_channels,
                                       kernel_size=original_conv1.kernel_size,
                                       stride=original_conv1.stride,
                                       padding=original_conv1.padding,
                                       bias=False if original_conv1.bias is None else True)
        num_ftrs = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet50(x)

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
    model_path = "resnet50_sentinel2_trained.pth"  # Path to your trained model file
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
    test_dataset = TestDatasetNPY(test_data_dir, label_map=label_map)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False) # No need to shuffle for evaluation

    # --- Load the Model ---
    model = ConvNeXtSentinel2(num_classes=num_classes)
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

    npy_file_path = "/Users/dom/projects/HSG-AIML-CC/testset/testset/test_1.npy"  # Replace with the actual path
    data = np.load(npy_file_path)
    print(f"Shape of {npy_file_path}: {data.shape}")
    test_data_dir = "./testset/testset"
    main()