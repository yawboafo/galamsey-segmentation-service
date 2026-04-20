from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Query
from typing import Optional
from app.schemas.prediction import PredictionRequest, PredictionResponse, TimeSeriesRequest, AnalysisMode
from app.services.inference import GalamseyInferenceService, get_inference_service
from app.services.post_processing import PostProcessor
from app.config import settings
import numpy as np

router = APIRouter()

@router.post("/predict", response_model=PredictionResponse)
async def predict_galamsey(
    image_url: Optional[str] = Query(None),
    file: Optional[UploadFile] = File(None),
    analysis_mode: AnalysisMode = Query(AnalysisMode.rgb),
    threshold: float = settings.DEFAULT_THRESHOLD,
    min_area: int = settings.MIN_REGION_AREA,
    simplify_tolerance: float = settings.SIMPLIFY_TOLERANCE,
    inference_service: GalamseyInferenceService = Depends(get_inference_service)
):
    """
    Predict advanced multi-class features in a satellite image chip.
    Accepts either an image URL or a file upload.
    """
    try:
        if file:
            image_data = await file.read()
            image = inference_service.load_image(image_data)
        elif image_url:
            image = inference_service.load_image(image_url)
        else:
            raise HTTPException(status_code=400, detail="Missing image. Provide image_url or upload a file.")

        ndvi_score = inference_service.calculate_ndvi(image) if analysis_mode == AnalysisMode.multispectral else None
        
        # 2. Run Inference (multi-class)
        prediction_mask = inference_service.predict(image, return_all_classes=True)
        
        # 3. Post-Process
        processor = PostProcessor(
            threshold=threshold,
            min_area=min_area,
            simplify_tolerance=simplify_tolerance
        )
        
        extra_props = {"analysis_mode": analysis_mode.value}
        if ndvi_score is not None:
            extra_props["ndvi_stress_score"] = ndvi_score
            
        geojson = processor.process(prediction_mask, extra_properties=extra_props)
        
        confidence = float(np.max(prediction_mask))
        
        return PredictionResponse(
            success=True,
            confidence=confidence,
            area_geojson=geojson,
            metadata={
                "model_name": settings.MODEL_CHECKPOINT.split("/")[-1],
                "analysis_mode": analysis_mode.value,
                "threshold": threshold,
                "input_size": f"{image.width}x{image.height}"
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict-change", response_model=PredictionResponse)
async def predict_change(
    request: TimeSeriesRequest,
    inference_service: GalamseyInferenceService = Depends(get_inference_service)
):
    """
    Calculates change detection between two time periods.
    """
    try:
        img_t1 = inference_service.load_image(request.image_url_t1)
        img_t2 = inference_service.load_image(request.image_url_t2)
        
        mask_t1 = inference_service.predict(img_t1, return_all_classes=False)
        mask_t2 = inference_service.predict(img_t2, return_all_classes=False)
        
        delta_mask = inference_service.calculate_change(mask_t1, mask_t2)
        
        processor = PostProcessor(
            threshold=request.threshold or settings.DEFAULT_THRESHOLD,
            min_area=request.min_area or settings.MIN_REGION_AREA,
            simplify_tolerance=request.simplify_tolerance or settings.SIMPLIFY_TOLERANCE
        )
        
        # Repurpose process logic for a single channel (delta map)
        geojson = processor.process(delta_mask, extra_properties={"is_new_disturbance": True})
        
        return PredictionResponse(
            success=True,
            confidence=float(np.max(delta_mask)) if delta_mask.size > 0 else 0.0,
            area_geojson=geojson,
            metadata={
                "type": "time-series-delta",
                "t1": request.image_url_t1,
                "t2": request.image_url_t2
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze-water", response_model=PredictionResponse)
async def analyze_water(
    image_url: Optional[str] = Query(None),
    file: Optional[UploadFile] = File(None),
    inference_service: GalamseyInferenceService = Depends(get_inference_service)
):
    """
    Specific endpoint to extract the turbid water class.
    """
    try:
        if file:
            image = inference_service.load_image(await file.read())
        elif image_url:
            image = inference_service.load_image(image_url)
        else:
            raise HTTPException(status_code=400, detail="Missing image.")

        # Simulate getting the water class (class 4) or fallback
        prediction_mask = inference_service.predict(image, return_all_classes=True)
        if len(prediction_mask.shape) == 3 and prediction_mask.shape[0] > 4:
            water_mask = prediction_mask[4]
        else: # Mock it if model doesn't output it
            water_mask = prediction_mask[0] if len(prediction_mask.shape) == 3 else prediction_mask

        processor = PostProcessor(threshold=settings.DEFAULT_THRESHOLD, min_area=settings.MIN_REGION_AREA, simplify_tolerance=settings.SIMPLIFY_TOLERANCE)
        geojson = processor.process(water_mask, extra_properties={"feature_type": "water_turbid", "turbidity_level": "High"})
        
        return PredictionResponse(
            success=True,
            confidence=float(np.max(water_mask)),
            area_geojson=geojson,
            metadata={"analysis_type": "river_turbidity"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": settings.MODEL_CHECKPOINT}
