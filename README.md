# MedScan AI

MedScan AI is an end-to-end medical image classification demo that predicts whether an uploaded image is **Disease** or **Normal**. It includes a PyTorch training pipeline, a FastAPI inference backend with Grad-CAM explainability, and a Next.js web interface for uploading images and viewing results.

> **Disclaimer:** This project is for **educational and portfolio purposes only**. It is not a medical device and must not be used for clinical diagnosis or treatment decisions.

## Features

- **Binary image classification** — ResNet50 transfer learning on a custom medical image dataset
- **Class imbalance handling** — inverse-frequency weighted loss during training
- **Explainable AI** — Grad-CAM heatmaps showing where the model focused for each prediction
- **Full-stack web app** — upload an image, get class probabilities, confidence score, and an attention map overlay
- **Reproducible pipeline** — separate train / validation / test splits with sklearn evaluation metrics

## Tech Stack

| Layer              | Technologies                                          |
| ------------------ | ----------------------------------------------------- |
| **ML / Training**  | Python, PyTorch, torchvision (ResNet50), scikit-learn |
| **Explainability** | pytorch-grad-cam                                      |
| **Backend**        | FastAPI, Uvicorn, Pillow                              |
| **Frontend**       | Next.js 16, React 19, HeroUI, Tailwind CSS v4         |

## Model Performance

Evaluated on the held-out **test set** (624 images) using the saved checkpoint at `models/medical_classifier.pth`:

| Metric                   | Score |
| ------------------------ | ----- |
| **Accuracy**             | 84.3% |
| **Precision** (weighted) | 84.2% |
| **Recall** (weighted)    | 84.3% |
| **F1-score** (weighted)  | 84.2% |

### Per-class results

| Class   | Precision | Recall | F1-score | Support |
| ------- | --------- | ------ | -------- | ------- |
| Disease | 0.86      | 0.89   | 0.88     | 390     |
| Normal  | 0.81      | 0.76   | 0.78     | 234     |

### Confusion matrix

Rows = true label, columns = predicted label (`Disease`, `Normal`):

```
[[349  41]
 [ 57 177]]
```

Normal recall is lower than Disease recall, which reflects class imbalance in training (~74% Disease / ~26% Normal). Class-weighted loss was used to partially mitigate this.

## Dataset

Images are organized in ImageFolder layout (not committed to Git — add locally):

```
data/
├── train/
│   ├── Disease/    # 3,875 images
│   └── Normal/     # 1,341 images
├── val/
│   ├── Disease/    # 23 images
│   └── Normal/     # 24 images
└── test/
    ├── Disease/    # 390 images
    └── Normal/     # 234 images
```

Supported formats: `.jpg`, `.jpeg`, `.png`
