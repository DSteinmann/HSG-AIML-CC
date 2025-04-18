import rasterio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import traceback

# --- Configuration ---
# !!! Adjust this path to point to your TRAINING data directory !!!
# Navigate into one of the class folders
TIF_TRAIN_DIR = Path('./ds/images/remote_sensing/otherDatasets/sentinel_2/tif/Forest') # Example class folder
# Pick one file to inspect
SAMPLE_TIF_FILE = TIF_TRAIN_DIR / 'Forest_21.tif' # Example file

print(f"Inspecting TIF file: {SAMPLE_TIF_FILE}")

if not SAMPLE_TIF_FILE.exists():
    print(f"ERROR: Sample TIF file not found at {SAMPLE_TIF_FILE}")
else:
    try:
        with rasterio.open(SAMPLE_TIF_FILE) as src:
            print(f"Number of bands found: {src.count}")

            if src.count != 13:
                 print(f"Warning: Expected 13 bands, found {src.count}")

            print("\nBand Descriptions/Indices (if available):")
            try:
                 # Print descriptions if they exist
                 if src.descriptions and len(src.descriptions) == src.count:
                     for i, desc in enumerate(src.descriptions):
                         print(f"  Band {i+1}: {desc}")
                 else:
                      print("  No detailed band descriptions found in metadata.")
                      # Just print indices
                      for i in src.indexes:
                          print(f"  Band index: {i}")

            except Exception as meta_e:
                print(f"  Could not read detailed metadata: {meta_e}")


            # --- Optional: Visualize all 13 bands ---
            print("\nVisualizing all bands found in TIF:")
            all_bands_data = src.read() # Reads all bands into (C, H, W)

            fig, axes = plt.subplots(3, 5, figsize=(20, 10)) # Adjust grid size if needed
            axes = axes.ravel()

            for i in range(src.count):
                if i < len(axes): # Avoid index error if more than 15 bands somehow
                    band_slice = all_bands_data[i, :, :]
                    title = f"TIF Band Index {i} (Band {i+1})"

                    vmin, vmax = np.percentile(band_slice[~np.isnan(band_slice)], [1, 99])
                    im = axes[i].imshow(band_slice, cmap='viridis', vmin=vmin, vmax=vmax)
                    axes[i].set_title(title)
                    axes[i].axis('off')
                    if vmin != vmax :
                       fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
                    else:
                       axes[i].text(0.5, 0.5, f'Uniform value: {vmin:.2f}', horizontalalignment='center', verticalalignment='center', transform=axes[i].transAxes)


            # Hide unused axes
            for j in range(src.count, len(axes)):
                 axes[j].axis('off')

            plt.tight_layout()
            plt.suptitle(f"Band Visualization for {SAMPLE_TIF_FILE.name}", y=1.02)
            plt.show()

    except Exception as e:
        print(f"An error occurred reading TIF: {e}")
        traceback.print_exc()