# setup_and_data_download.py

import subprocess
import os
import glob
import numpy as np
import rasterio as rio
from rasterio.plot import reshape_as_image
import matplotlib.pyplot as plt

def install_packages():
    try:
        subprocess.check_call(['pip', 'install', 'rasterio', 'matplotlib', 'kaggle'])
        print("Packages installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing packages: {e}")
        return
    except FileNotFoundError:
        print("Error: pip not found.  Make sure Python and pip are properly installed.")
        return

install_packages()

plt.switch_backend('Agg')

print("### Dataset Download")
print('<img align="center" style="max-width: 300px; height: auto" src="https://github.com/HSG-AIML-Teaching/ML2025-Lab/blob/main/cc_1/eurosat.png?raw=1">')
print("The Eurosat dataset is available on [github](https://github.com/phelber/EuroSAT).")
print("The multi-spectral (MS) version can be downloaded with the following command:")

# --- Download EuroSATallBands.zip (with check) ---
eurosat_zip_file = "EuroSATallBands.zip"
if not os.path.exists(eurosat_zip_file):
    try:
        subprocess.check_call(['wget', 'https://madm.dfki.de/files/sentinel/EuroSATallBands.zip', '--no-check-certificate'])
        print(f"{eurosat_zip_file} downloaded successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading {eurosat_zip_file}: {e}")
        exit()
    except FileNotFoundError:
        print("Error: wget not found. Please ensure wget is installed on your system.")
        exit()
else:
    print(f"{eurosat_zip_file} already exists. Skipping download.")

# --- Unzip EuroSATallBands.zip (with check) ---
eurosat_extracted_dir = "2750"  # Check for a common directory inside the zip
if not os.path.exists(eurosat_extracted_dir):
    try:
        subprocess.check_call(['unzip', '-q', eurosat_zip_file])
        print(f"{eurosat_zip_file} unzipped successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error unzipping {eurosat_zip_file}: {e}")
        exit()
    except FileNotFoundError:
        print("Error: unzip not found.  Make sure unzip is installed.")
        exit()
else:
    print(f"EuroSAT appears to be already extracted (found directory '{eurosat_extracted_dir}'). Skipping unzip.")
# --- Download Kaggle competition data (with check) ---
kaggle_zip_file = "8-860-1-00-coding-challenge-2025.zip"
if not os.path.exists(kaggle_zip_file):
    try:
        subprocess.check_call(['kaggle', 'competitions', 'download', '-c', '8-860-1-00-coding-challenge-2025'])
        print("Kaggle competition data downloaded successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading Kaggle data: {e}")
        print("Make sure you have the Kaggle API configured correctly.")
        exit()
    except FileNotFoundError:
        print("Error: kaggle command not found. Make sure the Kaggle CLI is installed.")
        exit()
else:
    print(f"{kaggle_zip_file} already exists. Skipping download.")

# --- Unzip the Kaggle competition data (with check) ---
kaggle_extracted_file = "test.zip" #check if test.zip exists. If so we consider it unzipped
if not os.path.exists(kaggle_extracted_file):
    try:
        subprocess.check_call(['unzip', '-q', kaggle_zip_file])
        print("Kaggle competition data unzipped successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error unzipping Kaggle competition data: {e}")
        exit()
    except FileNotFoundError:
        print("Error: unzip not found.  Make sure unzip is installed.")
        exit()
else:
    print("Kaggle competition data appears to be already unzipped. Skipping unzip.")
print("Data download and setup complete.")