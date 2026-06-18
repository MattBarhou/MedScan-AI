import base64
import io

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from services.model import get_device, get_model
from services.preprocessing import IMAGE_SIZE

HEATMAP_CAPTION = (
    "Highlighted areas show where the model focused when making this prediction. "
    "This is not a confirmed medical finding."
)


def generate_heatmap_base64(
    image_tensor: torch.Tensor,
    original_image: Image.Image,
    target_index: int,
) -> str:
    model = get_model()
    device = get_device()

    rgb_image = original_image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    rgb_array = np.float32(rgb_image) / 255.0

    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    input_tensor = image_tensor.to(device)
    targets = [ClassifierOutputTarget(target_index)]

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    heatmap = show_cam_on_image(rgb_array, grayscale_cam, use_rgb=True)

    buffer = io.BytesIO()
    Image.fromarray(heatmap).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return f"data:image/png;base64,{encoded}"


def get_heatmap_caption() -> str:
    return HEATMAP_CAPTION
