from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Same transforms used for validation and testing during training
_eval_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


def preprocess_image(image_path: Path) -> torch.Tensor:
    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")

    tensor = _eval_transform(rgb_image)
    return tensor.unsqueeze(0)
