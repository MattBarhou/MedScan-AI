import torch

from services.model import get_class_labels, get_device, get_model


def run_inference(image_tensor: torch.Tensor) -> dict:
    model = get_model()
    class_labels = get_class_labels()
    device = get_device()

    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    predicted_index = int(torch.argmax(probabilities).item())
    confidence = float(probabilities[predicted_index].item())

    probability_map = {
        label: round(float(probabilities[index].item()), 4)
        for index, label in enumerate(class_labels)
    }

    return {
        "predicted_class": class_labels[predicted_index],
        "predicted_index": predicted_index,
        "confidence": round(confidence, 4),
        "probabilities": probability_map,
    }
