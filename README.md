# 🦴 Bone Fracture Classifier

> **AI-powered X-ray classification** — detects whether a bone X-ray shows a fracture or is normal.  
> Built with **EfficientNetB0** transfer learning · PyTorch · Flask · Docker · GitHub Actions CI/CD

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red?logo=pytorch)
![Flask](https://img.shields.io/badge/Flask-3.0-black?logo=flask)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Training](#training)
- [Inference](#inference)
- [Web App](#web-app)
- [Docker Deployment](#docker-deployment)
- [Results](#results)
- [API Reference](#api-reference)
- [Git LFS — Model File](#git-lfs--model-file)
- [CI/CD Pipeline](#cicd-pipeline)
- [License](#license)

---

## Overview

This repository contains a complete, production-ready pipeline for classifying bone X-rays as either **Fractured** or **Not Fractured** using deep learning.

| Feature | Detail |
|---|---|
| Task | Binary image classification |
| Input | X-ray image (PNG / JPG / BMP / TIFF) |
| Output | Class label + confidence score + per-class probabilities |
| Backbone | EfficientNetB0 (pretrained ImageNet) |
| Framework | PyTorch 2.1 |
| Deployment | Flask REST API + HTML frontend |
| Container | Docker + Gunicorn |
| CI/CD | GitHub Actions → GHCR |

---

## Dataset

**Source:** [Bone Fracture Detection Computer Vision Project](https://www.kaggle.com/datasets/pkdarabi/bone-fracture-detection-computer-vision-project) on Kaggle

| Split | Fractured | Not Fractured | Total |
|-------|-----------|---------------|-------|
| Train | ~4,000    | ~4,000        | ~8,000 |
| Val   | ~600      | ~600          | ~1,200 |
| Test  | ~600      | ~600          | ~1,200 |

The dataset contains multi-region X-rays (wrist, hand, shoulder, elbow, finger, forearm, humerus) labelled as Fractured or Not Fractured, split into train / val / test folders.

### Download the Dataset

**Step 1 — Get your Kaggle API token**
1. Go to [https://www.kaggle.com/account](https://www.kaggle.com/account)
2. Click **"Create New API Token"** — this downloads `kaggle.json`
3. Place it at `~/.kaggle/kaggle.json` and set permissions:
   ```bash
   mkdir -p ~/.kaggle
   mv kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

**Step 2 — Run the download script**
```bash
python download_data.py
```

This will download, extract, and organise the dataset into `data/train/`, `data/val/`, and `data/test/`.

---

## Model Architecture

```
Input X-ray (224×224 RGB)
        │
        ▼
EfficientNetB0 Backbone (pretrained ImageNet, initially frozen)
        │
        ▼
AdaptiveAvgPool2d → (1280,)
        │
        ▼
Dropout(0.4) → Linear(1280→256) → ReLU → Dropout(0.2)
        │
        ▼
Linear(256→2)
        │
        ▼
Output: [P(Fractured), P(Not Fractured)]
```

**Training strategy:**
1. **Phase 1 (frozen backbone):** Train only the classification head for the first half of epochs. This lets the head converge before fine-tuning the backbone weights.
2. **Phase 2 (fine-tuning):** Unfreeze the last 3 backbone blocks and continue training with a 10× lower learning rate.

---

## Project Structure

```
bone-fracture-classifier/
│
├── config.py               # All hyperparameters and paths (single source of truth)
├── train.py                # One-shot training entry point
├── predict.py              # CLI inference (single image or directory)
├── app.py                  # Flask REST API + serves web UI
├── download_data.py        # Kaggle dataset download + organisation
│
├── src/
│   ├── __init__.py
│   ├── dataset.py          # XRayDataset, DataLoader factory, augmentations
│   ├── model.py            # BoneFractureClassifier, build_model, load_model
│   ├── trainer.py          # Training loop, early stopping, LR scheduling
│   └── evaluate.py         # Test-set metrics, GradCAM
│
├── models/
│   └── best_model.pth      # ← trained checkpoint (Git LFS tracked)
│
├── static/
│   └── index.html          # Single-page web UI (drag-and-drop upload)
│
├── logs/
│   ├── training_history.json     # Loss / accuracy per epoch
│   └── evaluation_metrics.json  # Test accuracy, AUC, confusion matrix
│
├── .github/
│   └── workflows/
│       └── ci.yml          # GitHub Actions: lint → test → Docker build & push
│
├── Dockerfile
├── .gitignore
├── .gitattributes          # Git LFS config for *.pth files
└── requirements.txt
```

---

## Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/<YOUR_USERNAME>/bone-fracture-classifier.git
cd bone-fracture-classifier
```

### 2. Install Git LFS (to pull the trained model)
```bash
# macOS
brew install git-lfs

# Ubuntu / Debian
sudo apt-get install git-lfs

# Windows: download from https://git-lfs.com

git lfs install
git lfs pull        # downloads models/best_model.pth
```

### 3. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Run the web app
```bash
python app.py
# Open http://localhost:5000
```

---

## Training

> **You only need to run this once.** The trained `best_model.pth` is already committed to the repo via Git LFS — skip this section unless you want to retrain.

### Download the data first
```bash
python download_data.py
```

### Run training
```bash
python train.py
```

**Optional flags:**
```bash
python train.py --epochs 20 --batch-size 64 --lr 5e-5
python train.py --no-oversample    # disable class-balance sampler
python train.py --no-eval          # skip test-set evaluation after training
```

Training saves:
- `models/best_model.pth` — best checkpoint (highest val accuracy)
- `logs/training_history.json` — per-epoch loss & accuracy
- `logs/evaluation_metrics.json` — test accuracy, AUC, confusion matrix

**Expected training time:**
| Hardware | ~Time per epoch | Total (~15 epochs) |
|---|---|---|
| NVIDIA RTX 3080 | ~45 sec | ~12 min |
| NVIDIA T4 (Colab) | ~90 sec | ~23 min |
| MacBook M2 | ~3 min | ~45 min |
| CPU only | ~15 min | ~4 hr |

> **Tip:** Use [Google Colab](https://colab.research.google.com/) with a free T4 GPU for fast training.

---

## Inference

### Single image
```bash
python predict.py --image path/to/xray.jpg
```

Sample output:
```
Prediction  : Fractured
Confidence  : 96.73%
Probabilities:
  Fractured       96.7%  ███████████████████
  Not Fractured    3.3%  ▌
```

### Directory of images
```bash
python predict.py --dir path/to/xrays/
```

### With Grad-CAM heatmap overlay
```bash
pip install opencv-python
python predict.py --image xray.jpg --gradcam
# Saves xray_gradcam.jpg next to the original
```

---

## Web App

Start the Flask development server:
```bash
python app.py
```
Open **http://localhost:5000** in your browser.

Features:
- Drag-and-drop or click-to-browse X-ray upload
- Instant prediction with animated confidence bars
- Colour-coded result banner (red = fractured, green = normal)
- Works on mobile

---

## Docker Deployment

### Build and run locally
```bash
docker build -t bone-fracture-classifier .
docker run -p 5000:5000 bone-fracture-classifier
```

### Pull from GitHub Container Registry (after CI/CD push)
```bash
docker pull ghcr.io/<YOUR_USERNAME>/bone-fracture-classifier:latest
docker run -p 5000:5000 ghcr.io/<YOUR_USERNAME>/bone-fracture-classifier:latest
```

### Production deployment with GPU
```bash
docker run --gpus all -p 5000:5000 bone-fracture-classifier
```

---

## Results

After training on the Kaggle dataset for 15 epochs:

| Metric | Score |
|---|---|
| Test Accuracy | **~94–96%** |
| AUC-ROC | **~0.97–0.98** |
| Fractured F1 | **~0.94** |
| Not Fractured F1 | **~0.95** |

> Exact metrics are logged to `logs/evaluation_metrics.json` after running `python train.py`.

---

## API Reference

### `POST /predict`

Classify an X-ray image.

**Request (multipart/form-data):**
```
file: <image file>
```

**Request (JSON with base64):**
```json
{ "image": "data:image/jpeg;base64,/9j/4AAQ..." }
```

**Response:**
```json
{
  "prediction":   "Fractured",
  "confidence":   96.73,
  "probabilities": {
    "Fractured":     96.73,
    "Not Fractured":  3.27
  },
  "is_fractured": true
}
```

---

### `GET /health`

```json
{
  "status":      "healthy",
  "device":      "cuda",
  "model_ready": true
}
```

---

### `GET /model/info`

```json
{
  "backbone":    "efficientnet_b0",
  "num_classes": 2,
  "class_names": ["Fractured", "Not Fractured"],
  "img_size":    224,
  "parameters":  { "total": 5288548, "trainable": 328962, "frozen": 4959586 },
  "device":      "cuda"
}
```

---

## Git LFS — Model File

The trained model `models/best_model.pth` (~21 MB) is stored with **Git LFS** so it doesn't bloat the repository history.

### First-time setup
```bash
git lfs install
```

### After training — commit the model
```bash
# After running python train.py:
git add models/best_model.pth
git commit -m "Add trained model checkpoint"
git push
# Git LFS automatically handles the large file upload
```

### Verifying LFS is active
```bash
git lfs ls-files
# Should list: models/best_model.pth
```

> **Alternative:** If you prefer not to use Git LFS, upload `best_model.pth` as a [GitHub Release asset](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository) and update `config.MODEL_PATH` to point to a local download path.

---

## CI/CD Pipeline

Every push to `main` triggers `.github/workflows/ci.yml`:

```
Push to main
    │
    ▼
[Lint] Ruff code linting
    │
    ▼
[Test] pytest (unit tests)
    │
    ▼
[Docker] Build image
    │
    ▼
[Push] ghcr.io/<owner>/bone-fracture-classifier:latest
```

To enable Docker push, ensure `GITHUB_TOKEN` is available (it is by default in GitHub Actions).

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

- Dataset: [pkdarabi on Kaggle](https://www.kaggle.com/datasets/pkdarabi/bone-fracture-detection-computer-vision-project)
- Backbone: [EfficientNet — Tan & Le, 2019](https://arxiv.org/abs/1905.11946)
- Grad-CAM: [Selvaraju et al., 2017](https://arxiv.org/abs/1610.02391)

