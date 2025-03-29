import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from Res2Model import Sentinel2Dataset

if __name__ == '__main__':
    root_directory = "./ds/images/remote_sensing/otherDatasets/sentinel_2/tif"  # Replace with your actual training data path
    image_size = 256  # Or whatever size you are resizing to
    num_workers = 0  # Start with 0 to avoid shared memory issues for calculation
    batch_size = 64  # Adjust this batch size based on your memory capacity

    # Create the training dataset (without any normalization transforms yet)
    train_dataset = Sentinel2Dataset(root_directory, transform=transforms.Resize((image_size, image_size)))

    # Create a DataLoader to load the entire dataset (or a large enough sample)
    # Set batch_size to a large number to process as much data as possible at once
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    num_bands = 12  # Number of channels in your Sentinel-2 data (after removing the 10th band)
    total_sum = torch.zeros(num_bands)
    total_squared_sum = torch.zeros(num_bands)
    total_pixels = 0

    print("Calculating mean and standard deviation...")

    for batch_idx, (data, _) in enumerate(dataloader):
        if data is not None:
            # Data will have shape (batch_size, num_channels, height, width)
            data = data.float()  # Ensure data is float for calculations
            total_sum += torch.sum(data, dim=(0, 2, 3))
            total_squared_sum += torch.sum(data ** 2, dim=(0, 2, 3))
            total_pixels += data.numel() // num_bands

    if total_pixels > 0:
        train_mean = total_sum / total_pixels
        train_std = torch.sqrt((total_squared_sum / total_pixels) - (train_mean ** 2))

        print(f"Calculated Mean: {train_mean.tolist()}")
        print(f"Calculated Standard Deviation: {train_std.tolist()}")
    else:
        print("Error: No data found in the training dataset.")