import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import traceback

# --- Configuration ---
# !!! Adjust this path to point to your testset directory !!!
NPY_TEST_DIR = Path('./testset/testset')
# Pick one file to inspect
SAMPLE_NPY_FILE = NPY_TEST_DIR / 'test_0.npy'  # Or any other test file

# Define expected standard band names for the 12 L2A channels for reference
# (Order: B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B11, B12)
EXPECTED_STANDARD_BAND_NAMES = [
    "B1_Coastal", "B2_Blue", "B3_Green", "B4_Red",
    "B5_RE1", "B6_RE2", "B7_RE3", "B8_NIR",
    "B8A_NIR2", "B9_WV", "B11_SWIR1", "B12_SWIR2"
]

# --- Load and Inspect ---
print(f"Inspecting file: {SAMPLE_NPY_FILE}")

if not SAMPLE_NPY_FILE.exists():
    print(f"ERROR: Sample file not found at {SAMPLE_NPY_FILE}")
else:
    try:
        # Load the NPY data
        image_data = np.load(SAMPLE_NPY_FILE)
        print(f"Loaded NPY data shape: {image_data.shape}")

        # Transpose if it looks like (H, W, C) instead of (C, H, W)
        if image_data.ndim == 3 and image_data.shape[0] != 12 and image_data.shape[2] == 12:
            print("Transposing from (H, W, C) to (C, H, W)")
            image_data = image_data.transpose(2, 0, 1)

        # Check final shape
        if image_data.ndim != 3 or image_data.shape[0] != 12:
            print(f"ERROR: Unexpected data shape after potential transpose: {image_data.shape}. Expected (12, H, W)")
        else:
            print(f"Data shape is (C, H, W): {image_data.shape}")
            num_bands, height, width = image_data.shape

            # --- Visualization (Optional but helpful) ---
            print("\nVisualizing selected bands (assuming standard order for plot titles):")
            fig, axes = plt.subplots(3, 4, figsize=(15, 10))
            axes = axes.ravel()  # Flatten the grid

            for i in range(num_bands):
                band_slice = image_data[i, :, :]
                title = f"Index {i}: {EXPECTED_STANDARD_BAND_NAMES[i]} (?)"  # Title based on STANDARD order assumption

                # Handle potential outliers for visualization clipping
                vmin, vmax = np.percentile(band_slice[~np.isnan(band_slice)], [1, 99])
                im = axes[i].imshow(band_slice, cmap='viridis', vmin=vmin, vmax=vmax)
                axes[i].set_title(title)
                axes[i].axis('off')
                # Add colorbar only if values are not uniform
                if vmin != vmax:
                    fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
                else:
                    axes[i].text(0.5, 0.5, f'Uniform value: {vmin:.2f}', horizontalalignment='center',
                                 verticalalignment='center', transform=axes[i].transAxes)

            plt.tight_layout()
            plt.suptitle(
                f"Band Visualization for {SAMPLE_NPY_FILE.name}\n(Titles assume standard B1-B12 order; check if visuals match band type)",
                y=1.02)
            plt.show()

            # --- Print Basic Stats ---
            print("\nBasic Statistics per Band Index:")
            print("Index | Assumed Std Name |   Min    |   Max    |   Mean   |   Std Dev")
            print("------|------------------|----------|----------|----------|----------")
            for i in range(num_bands):
                band_slice = image_data[i, :, :].astype(np.float32)  # Ensure float for calculations
                band_name = EXPECTED_STANDARD_BAND_NAMES[i]
                min_val = np.nanmin(band_slice)
                max_val = np.nanmax(band_slice)
                mean_val = np.nanmean(band_slice)
                std_val = np.nanstd(band_slice)
                print(f"{i:^6}| {band_name:<16} | {min_val:8.2f} | {max_val:8.2f} | {mean_val:8.2f} | {std_val:8.2f}")

    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()