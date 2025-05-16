# HSG-AIML-CC


## Project Overview

This repository provides code and resources for classifying **land use and land cover** from **Sentinel-2 satellite images** using the deep learning approaches we have learned in class. Leveraging the **EuroSAT dataset** (from the Copernicus program), we implement and experiment with **Convolutional Neural Networks (CNNs)** to recognize 10 land cover classes with pre-trained and non-pretrained weights. The core aim is to analyze the impact of different spectral band combinations and model architectures on classification accuracy in an explorative manner.
---
## Dataset

We use the [EuroSAT dataset](https://github.com/phelber/eurosat), which features:
- Around 27,000 images
- 10 classes of images: Residential, Industrial, River, Forest, Pasture, Annual Crop, Permanent Crop, Herbaceous Vegetation, Highway, Sea & Lake
- 13 spectral bands from Sentinel-2 as well as some additional indices to increase classification accuracy

### Download

- Download directly from [EuroSAT GitHub](https://github.com/phelber/eurosat)
- We used EuroSAT (MS)
---

## Model & Methodology

We used a CNN architecture (ResNet50) and experimented with (1) several parameters which can be found in the config_XX.yaml-files for both approaches (non-pretrained = track 1 and pre-trained = track 2) as well as (2) indices (NDVI, NDWI, NDBI and NDRE1) and (3) image-transformation methods.

---

## Repository Structure & File Descriptions

| Name                          | Type      | Description |
|-------------------------------|-----------|-------------|
| `.gitignore`                  | file      | Specifies files/folders ignored by git (e.g., data, outputs, environments). |
| `README.md`                   | file      | Project documentation and usage instructions (this file). |
| `cc_01_getting_started.ipynb` | notebook  | Jupyter Notebook introducing the project, setup, and basic usage. |
| `checkBands.py`               | script    | Utility script for inspecting/validating spectral bands in dataset images. |
| `config_eval_track1.yaml`     | config    | Evaluation configuration for experimental "track 1". |
| `config_eval_track2.yaml`     | config    | Evaluation configuration for "track 2". |
| `config_track1.yaml`          | config    | Training/experiment configuration for "track 1". |
| `config_track2.yaml`          | config    | Training/experiment configuration for "track 2". |
| `eval.py`                     | script    | Script for evaluating model performance and generating metrics/reports. |
| `train.py`                    | script    | Main training pipeline for CNNs: data loading, training, checkpointing. |
| `old/`                        | folder    | Legacy scripts, deprecated code, or earlier experiment versions. |

---

## Getting Started

1. **Clone the repository**
   ```sh
   git clone https://github.com/DSteinmann/HSG-AIML-CC.git
   cd HSG-AIML-CC
   ```

2. **Set up the environment**
   - Install dependencies:
     ```sh
     pip install -r requirements.txt
     ```
3. **Prepare the data**
   - Download the EuroSAT dataset and place it into the according path to your data.
4. **Run experiments**
---

## Example Usage

**Training:**
```sh
python train.py --config config_track1.yaml
```

**Evaluation:**
```sh
python eval.py --config config_eval_track1.yaml
```

---

## Notebooks

- **cc_01_getting_started.ipynb**: Interactive introduction, setup, basic data exploration, and experiment guide.

---

## Configuration Files

- **config_track1.yaml / config_track2.yaml**: Model training settings (hyperparameters, input bands, dataset splits).
- **config_eval_track1.yaml / config_eval_track2.yaml**: Evaluation parameters for trained models.

---

## Utility Scripts

- **checkBands.py**: Inspect/validate image spectral bands before training.

---

## Legacy Code

- **old/**: Deprecated scripts and previous experiment code.

---

## References

- [EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification](https://arxiv.org/abs/1709.00029)
- [EuroSAT GitHub](https://github.com/phelber/eurosat)

---
## Directory of Aids 

| Aid                          | Description      | Areas Affected |
|-------------------------------|-----------|-------------|
| `Google's Gemini AI`                  | X     | X |
| `ChatGPT4o`                   | X      | X |
| `Github Copilot` | X  | X |
