from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Query
from typing import Optional
from app.schemas.prediction import PredictionRequest, PredictionResponse
from app.services.inference import GalamseyInferenceService, get_inference_service
from app.services.post_processing import PostProcessor
from app.config import settings
import numpy as np

router = APIRouter()

@router.post("/predict", response_model=PredictionResponse)
async def predict_galamsey(
    image_url: Optional[str] = Query(None),
    file: Optional[UploadFile] = File(None),
    threshold: float = settings.DEFAULT_THRESHOLD,
    min_area: int = settings.MIN_REGION_AREA,
    simplify_tolerance: float = settings.SIMPLIFY_TOLERANCE,
    inference_service: GalamseyInferenceService = Depends(get_inference_service)
):
    """
    Predict galamsey footprints in a satellite image chip.
    Accepts either an image URL or a file upload.
    """
    try:
        # 1. Load Image
        if file:
            image_data = await file.read()
            image = inference_service.load_image(image_data)
        elif image_url:
            image = inference_service.load_image(image_url)
        else:
            raise HTTPException(status_code=400, detail="Missing image. Provide image_url or upload a file.")

        # 2. Run Inference
        prediction_mask = inference_service.predict(image)
        
        # 3. Post-Process
        processor = PostProcessor(
            threshold=threshold,
            min_area=min_area,
            simplify_tolerance=simplify_tolerance
        )
        geojson = processor.process(prediction_mask)
        
        # 4. Calculate overall confidence (mean of detected galamsey pixels or max)
        # For simplicity, we'll return the max confidence seen in the mask
        confidence = float(np.max(prediction_mask))
        
        return PredictionResponse(
            success=True,
            confidence=confidence,
            area_geojson=geojson,
            metadata={
                "model_name": settings.MODEL_CHECKPOINT.split("/")[-1],
                "threshold": threshold,
                "input_size": f"{image.width}x{image.height}"
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": settings.MODEL_CHECKPOINT}
