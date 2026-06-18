from pathlib import Path

import torch
import torch.nn as nn
from torchvision import datasets, models
from torchvision.models import ResNet50_Weights

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "medical_classifier.pth"
TRAIN_DIR = PROJECT_ROOT / "data" / "train"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model = None
_class_labels = None


def get_device() -> torch.device:
    return DEVICE


def _load_class_labels_from_data() -> list[str]:
    """Use the same ImageFolder class order as training."""
    if TRAIN_DIR.exists():
        dataset = datasets.ImageFolder(TRAIN_DIR)
        return dataset.classes

    return ["Disease", "Normal"]


def _load_checkpoint() -> dict:
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint

    return {"state_dict": checkpoint, "class_names": _load_class_labels_from_data()}


def build_model(num_classes: int) -> nn.Module:
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def initialize_model() -> None:
    """Load the trained model once at app startup."""
    global _model, _class_labels

    if _model is not None:
        return

    if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size == 0:
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. "
            "Run `python training/train.py` first."
        )

    checkpoint = _load_checkpoint()
    state_dict = checkpoint["state_dict"]
    _class_labels = checkpoint.get("class_names") or _load_class_labels_from_data()

    model = build_model(len(_class_labels))
    model.load_state_dict(state_dict)
    model.eval()
    _model = model.to(DEVICE)


def get_model() -> nn.Module:
    if _model is None:
        raise RuntimeError("Model is not loaded. The app must call initialize_model() on startup.")

    return _model


def get_class_labels() -> list[str]:
    if _class_labels is None:
        raise RuntimeError("Model is not loaded. The app must call initialize_model() on startup.")

    return _class_labels
