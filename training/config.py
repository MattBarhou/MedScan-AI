from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = "data"
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
TEST_DIR = "data/test"
MODEL_SAVE_PATH = "models/medical_classifier.pth"

CLASS_NAMES = ["Disease", "Normal"]
IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_EPOCHS = 13
LEARNING_RATE = 0.001

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def project_path(relative_path: str) -> Path:
    return PROJECT_ROOT / relative_path
