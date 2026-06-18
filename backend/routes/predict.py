import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from services.explainability import generate_heatmap_base64, get_heatmap_caption
from services.inference import run_inference
from services.preprocessing import preprocess_image
from utils.validation import validate_image_file

router = APIRouter()

TEMP_DIR = Path(__file__).resolve().parent.parent / "temp"
DISCLAIMER = (
    "This prediction is for educational purposes only and is not a medical diagnosis."
)


@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    validate_image_file(file)

    TEMP_DIR.mkdir(exist_ok=True)

    extension = "." + file.filename.rsplit(".", 1)[-1].lower()
    temp_path = TEMP_DIR / f"{uuid.uuid4()}{extension}"

    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        temp_path.write_bytes(contents)

        try:
            with Image.open(temp_path) as image:
                image.verify()
        except (UnidentifiedImageError, OSError) as exc:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is not a valid image.",
            ) from exc

        try:
            with Image.open(temp_path) as image:
                original_image = image.convert("RGB")

            image_tensor = preprocess_image(temp_path)
            prediction = run_inference(image_tensor)
            heatmap_base64 = generate_heatmap_base64(
                image_tensor,
                original_image,
                prediction["predicted_index"],
            )
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail="Prediction failed. Please try again with a different image.",
            ) from exc

        return {
            "filename": file.filename,
            "predicted_class": prediction["predicted_class"],
            "confidence": prediction["confidence"],
            "probabilities": prediction["probabilities"],
            "heatmap_base64": heatmap_base64,
            "heatmap_caption": get_heatmap_caption(),
            "disclaimer": DISCLAIMER,
        }
    finally:
        if temp_path.exists():
            temp_path.unlink()
