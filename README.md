<p align="center">
  <h1 align="center">Public Showcase: HSI Drill Core Classification</h1>
  <p align="center">
    <em>A deep learning pipeline for automated lithological mapping from hyperspectral core scans.</em>
  </p>
</p>

> **Note:** This is a public showcase version of my thesis code. The novel proprietary architecture (Multi-Scale CNN) has been replaced with a standard baseline model to protect intellectual property. However, this repository demonstrates my end-to-end machine learning pipeline, including data engineering, training, and evaluation.

<p align="center">
  <a href="https://colab.research.google.com/github/suva1998/hsi-lithology-showcase/blob/main/notebooks/01_data_exploration.ipynb">
    <img src="https://img.shields.io/badge/Colab-01_Data_Exploration-F9AB00?logo=googlecolab&logoColor=white" alt="01 — Data Exploration">
  </a>
  <a href="https://colab.research.google.com/github/suva1998/hsi-lithology-showcase/blob/main/notebooks/02_preprocessing_pipeline.ipynb">
    <img src="https://img.shields.io/badge/Colab-02_Preprocessing-F9AB00?logo=googlecolab&logoColor=white" alt="02 — Preprocessing">
  </a>
  <a href="https://colab.research.google.com/github/suva1998/hsi-lithology-showcase/blob/main/notebooks/03_train_baseline_cnn.ipynb">
    <img src="https://img.shields.io/badge/Colab-03_Train_Model-F9AB00?logo=googlecolab&logoColor=white" alt="03 — Train Baseline CNN">
  </a>
  <br>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python 3.9+"></a>
  <a href="https://www.tensorflow.org/"><img src="https://img.shields.io/badge/TensorFlow-2.15+-orange.svg" alt="TensorFlow 2.15+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
</p>

---

## Table of Contents

- [Context](#context)
- [Architecture](#architecture)
- [Key Design Decisions](#key-design-decisions)
- [Results](#results)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Limitations & Future Work](#limitations--future-work)
- [License](#license)
- [Author](#author)

---

## Context

> **From Master's Thesis to Open-Source Portfolio**
>
> This project originated from my master's thesis, where I developed a Multi-Scale CNN architecture that successfully classified **5 distinct macro-lithologies** from private, NDA-restricted hyperspectral drill core data acquired by DMT Group.
>
> To make this work **open-source and reproducible**, I adapted the exact same architecture to train on the publicly available **[HZDR RODARE Upscaling Mineralogy Benchmark](https://rodare.hzdr.de/)** (Thiele et al., 2024). This dataset provides co-registered hyperspectral imagery (FENIX VNIR-SWIR sensor, 450 bands) and high-resolution SEM-MLA mineral abundance ground truth across multiple European drill core collections.
>
> The architecture's multi-scale feature extraction capability was validated against ResNet50, MobileNetV3Large, EfficientNetB0, EfficientNetV2S, and EfficientNetB3 baselines in the thesis. This repository focuses on demonstrating the complete pipeline — from raw ENVI hyperspectral cubes to trained mineral classifier — on the Stonepark sub-dataset.

---

## Architecture

The system utilizes a deep learning pipeline designed to process hyperspectral drill core data and output mineralogical classifications.

```text
                 ┌────────────────────────────────────┐
                 │      Hyperspectral Input Cube      │
                 │    (Preprocessed & Normalized)     │
                 └─────────────────┬──────────────────┘
                                   │
                 ┌─────────────────▼──────────────────┐
                 │   Spatial-Spectral Feature Ext.    │
                 │   (e.g., Convolutional Backbone)   │
                 └─────────────────┬──────────────────┘
                                   │
                 ┌─────────────────▼──────────────────┐
                 │     Multi-Scale Feature Fusion     │
                 │   (Hierarchical context capture)   │
                 └─────────────────┬──────────────────┘
                                   │
                 ┌─────────────────▼──────────────────┐
                 │      Lithology Classification      │
                 │      (Dense layers + Softmax)      │
                 └────────────────────────────────────┘
```

**Note:** The exact structural details of the Multi-Scale CNN (including specific dilation rates, layer dimensions, and fusion mechanisms) are proprietary to the thesis research and have been abstracted in this diagram, and replaced with a baseline model in the code.

---

## Key Design Decisions

| Decision | Choice | Why |
|---|---|---|
| **Dimensionality reduction** | FastICA (30 components) | ICA separates statistically *independent* mineral absorption signatures. PCA maximizes variance, which often captures illumination — not mineralogy. Du et al. (2003) showed 15–20% accuracy gains with ICA for mineral classification. |
| **Architecture** | Baseline CNN | A standard custom CNN implemented as a structural placeholder. |
| **Loss function** | Class-balanced focal loss (γ=1.5) | Geological datasets have extreme class imbalance (rare minerals). Focal loss down-weights easy/majority samples; class-balanced α weights use the effective number formula (Cui et al., 2019). |
| **Augmentation** | MixUp (α=0.2) + geometric (flip/rot90) | Drill cores have no fixed spatial orientation → rotations are geologically valid. MixUp creates virtual examples that smooth decision boundaries and reduce overconfidence. |
| **Patch size** | 25×25 pixels | Captures sufficient spatial context around each labeled pixel while staying within the spatial extent of individual thin-sections. |

---

## Results

> **Note:** The results and metrics below (e.g., 96.0% test accuracy) were achieved using the proprietary **Multi-Scale CNN architecture**, tested exclusively on this **Stonepark sub-dataset**. The Baseline CNN provided in this public repository serves as a structural placeholder for the data pipeline and will yield different performance metrics.

| Metric | Multi-Scale CNN (Proprietary) | Baseline CNN (Open Source) |
|---|---|---|
| **Test Accuracy** | 96.0% | 92.0% |
| **Best Validation Accuracy** | 97.7% | 94.4% |
| **Best Validation Loss** | 0.0021 | 0.0165 |
| **F1-Score (Macro)** | 0.95 | 0.92 |

**Per-Class Performance** (test set):

| Mineral Class | Test Samples | Precision (Multi-Scale) | Recall (Multi-Scale) | F1-Score (Multi-Scale) | Precision (Baseline) | Recall (Baseline) | F1-Score (Baseline) |
|---|---|---|---|---|---|---|---|
| Sulphate | 84 | 0.79 | 0.96 | 0.87 | 0.76 | 0.89 | 0.82 |
| Muscovite | 300 | 0.99 | 0.96 | 0.98 | 0.97 | 0.92 | 0.94 |
| TiOx | 6 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

> **Note on class count**: The full MLA taxonomy contains 20+ mineral macro-groups. However, geology is *spatially clustered* — individual drill core sections are dominated by the local lithology. The Stonepark subset contains cores where Sulphate, Muscovite, and TiOx are the dominant minerals. The architecture handles the full class count dynamically (`num_classes` is config-driven), and was validated on 5 lithologies in the original thesis dataset.

---

## Quick Start

### Option 1 — Google Colab (Recommended)

Run the full pipeline in your browser with zero setup:

1. **Explore the data** — visualize drill core HSI cubes and MLA ground truth maps:

   [![01 — Data Exploration](https://img.shields.io/badge/Colab-01_Data_Exploration-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/suva1998/hsi-lithology-showcase/blob/main/notebooks/01_data_exploration.ipynb)

2. **Preprocess the data** — normalize, apply FastICA, and extract spatial-spectral patches:

   [![02 — Preprocessing](https://img.shields.io/badge/Colab-02_Preprocessing-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/suva1998/hsi-lithology-showcase/blob/main/notebooks/02_preprocessing_pipeline.ipynb)

3. **Train the model** — run the Baseline CNN training loop with focal loss:

   [![03 — Train Baseline CNN](https://img.shields.io/badge/Colab-03_Train_Model-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/suva1998/hsi-lithology-showcase/blob/main/notebooks/03_train_baseline_cnn.ipynb)

### Option 2 — Local Installation

```bash
# Clone the repository
git clone https://github.com/suva1998/hsi-lithology-showcase.git
cd hsi-lithology-showcase

# Install in editable mode
pip install -e .
```

**Train programmatically:**

```python
from src.data import load_envi_dataset, preprocess_hsi
from src.models import build_baseline_cnn
from src.training import Trainer
import yaml

# Load configuration
config = yaml.safe_load(open("configs/mla_drillcore.yaml"))

# Load raw ENVI data and preprocess
data, labels, class_names = load_envi_dataset("data/raw/MLA")
result = preprocess_hsi(data, labels, config)

# Train the model
trainer = Trainer(config)
history = trainer.train(result["splits"], result["class_counts"])
```

---

## Project Structure

```
hsi-lithology-showcase/
│
├── configs/
│   └── mla_drillcore.yaml          # All hyperparameters (nothing hardcoded)
│
├── src/
│   ├── data/
│   │   ├── download.py             # ENVI format data loading (FENIX + MLA)
│   │   ├── preprocessing.py        # Normalize → FastICA → patch extraction → split
│   │   └── augmentation.py         # MixUp + geometric augmentation (tf.data)
│   │
│   ├── models/
│   │   └── baseline_cnn.py         # Standard CNN placeholder
│   │
│   ├── training/
│   │   ├── trainer.py              # Training loop, checkpointing, cross-validation
│   │   ├── losses.py               # Focal loss + class-balanced weighting
│   │   └── scheduler.py            # Cosine decay with linear warmup
│   │
│   └── evaluation/
│       ├── metrics.py              # OA, AA, κ, F1, confusion matrix
│       └── visualization.py        # Publication-quality plots
│
├── notebooks/
│   ├── 01_data_exploration.ipynb   # Visualize HSI cubes + MLA ground truth
│   ├── 02_preprocessing_pipeline.ipynb  # End-to-end preprocessing demo
│   └── 03_train_baseline_cnn.ipynb      # Full training + evaluation
│
├── data/                           # gitignored — downloaded/generated at runtime
│   ├── raw/                        # Raw ENVI files from HZDR RODARE
│   └── processed/                  # Preprocessed patches (auto-generated)
│
├── experiments/results/            # Saved plots and classification reports
├── tests/                          # Unit tests for model and preprocessing
├── requirements.txt                # Pinned dependencies
└── setup.py                        # Editable install (pip install -e .)
```

---

## Dataset

This project uses the **Upscaling Mineralogy Benchmark** from the Helmholtz-Zentrum Dresden-Rossendorf (HZDR):

| Property | Details |
|---|---|
| **Source** | [HZDR RODARE](https://rodare.hzdr.de/) (Thiele et al., 2024) |
| **Sensor** | FENIX VNIR-SWIR (378–2502 nm, 450 bands) |
| **Ground truth** | SEM-based Mineral Liberation Analysis (MLA) |
| **Format** | ENVI standard (BSQ interleave, float32) |
| **Drill sites** | BVR, Bosenbrunn, Collinstown, Freiberg, Geyer, Spremberg, Stonepark |
| **Thin-sections** | 204 across 49 boreholes |
| **Mineral classes** | 30 raw minerals → 20 macro-groups via abundance mapping |

### Data Setup (Local)

> **Google Colab users**: The notebooks handle data download automatically — no manual setup needed.

For **local runs**, download the Stonepark subset from [HZDR RODARE](https://rodare.hzdr.de/) and place it under `data/raw/` in the repository root:

```
hsi-lithology-showcase/
└── data/
    └── raw/
        └── MLA/
            ├── AbundanceMapping.xlsx       # Mineral-to-macro-group mapping table
            └── Stonepark.hyc/              # Drill site folder
                ├── bG11A1.hyc/             # Individual thin-section
                │   ├── FENIX.dat/.hdr      # Hyperspectral cube (H × W × 450)
                │   ├── MLA_HSI.dat/.hdr    # Mineral abundance map (H × W × 30)
                │   ├── mask.dat/.hdr       # Valid pixel mask
                │   └── RGB.dat/.hdr        # True-color reference image
                ├── bG11A2.hyc/
                └── ...
```

The `data/` directory is **gitignored** and will never be committed. The loading pipeline (`src/data/download.py`) automatically discovers all `.hyc` directories, reads ENVI files via the `spectral` library, and maps fractional mineral abundances to discrete macro-group labels using `AbundanceMapping.xlsx`.

---

## Methodology

### Preprocessing Pipeline

```
Raw HSI Cube (H × W × 450 bands)
        │
        ▼
  1. Standard Scaling (per-band zero mean, unit variance)
        │
        ▼
  2. FastICA → 30 independent components
     (separates mineral absorption signatures)
        │
        ▼
  3. RGB Composite: IC1→R, IC2→G, IC3→B
     (false-color for Baseline CNN)
        │
        ▼
  4. Patch Extraction: 25×25 centered on labeled pixels
     (zero-padded at borders)
        │
        ▼
  5. Stratified Train/Val/Test Split (70/10/20)
```

### Training Protocol

- **Epochs**: 150 (Batch size: 64)
- **Optimizer**: Adam (Weight decay: 1e-4)
- **LR Schedule**: Cosine Decay with Linear Warmup
  - *Warmup phase*: 10 epochs ramping up to 1e-3
  - *Decay phase*: 140 epochs decaying to 1e-6
- **Loss**: Class-balanced focal loss (γ=2.0) using inverse frequency weighting
- **Augmentation**: MixUp (α=0.2) + geometric (flip/rot90)
- **Early stopping**: Monitors `val_loss` with patience=20
- **Cross-validation**: Stratified 5-fold CV available via `Trainer.cross_validate()`

---

## Limitations & Future Work

- **Public dataset simplicity**: The Stonepark subset contains only 3 dominant mineral classes due to spatial geological clustering. The architecture's multi-scale advantage is more pronounced on datasets with greater spatial complexity and class diversity (as demonstrated in the thesis on 5-class DMT data).
- **Small spatial extent**: Individual thin-sections are ~24×39 pixels, making large patches redundant. Adapting the patch size and dilation rates to the spatial resolution could improve efficiency.
- **Spectral utilization**: The current pipeline reduces 450 bands to 3 RGB channels for the Baseline CNN. A custom 1D-spectral + 2D-spatial dual-branch architecture could leverage the full spectral information without this bottleneck.
- **Generalization across sites**: Training on one drill site (Stonepark) and testing on another (e.g., Freiberg) would be a more rigorous evaluation of geological transfer learning.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Author

**Suvarup Ghosh** — Data Scientist & ML Engineer

- 📧 [suvarupghosh7916@gmail.com](mailto:suvarupghosh7916@gmail.com)
- 🔗 [LinkedIn](https://www.linkedin.com/in/suvarupghosh-56784a184)
- 💻 [GitHub](https://github.com/suva1998)
