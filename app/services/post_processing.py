import numpy as np
from app.utils.geo import mask_to_geojson, clean_mask
from app.config import settings
from typing import Dict, Any

class PostProcessor:
    def __init__(
        self, 
        threshold: float = settings.DEFAULT_THRESHOLD,
        min_area: int = settings.MIN_REGION_AREA,
        simplify_tolerance: float = settings.SIMPLIFY_TOLERANCE
    ):
        self.threshold = threshold
        self.min_area = min_area
        self.simplify_tolerance = simplify_tolerance

    def process(self, prediction_mask: np.ndarray, extra_properties: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Takes raw probability mask (can be [H, W] or [C, H, W]) and converts to cleaned GeoJSON.
        Class mapping (if C > 1):
        0: background, 1: galamsey_pit, 2: vegetation_loss, 3: road, 4: water_turbid
        """
        class_names = {
            1: "galamsey_pit",
            2: "vegetation_loss",
            3: "road",
            4: "water_turbid"
        }
        
        all_features = []
        
        # If single channel [H, W], wrap it to behave like [1, H, W] representing class 1
        if len(prediction_mask.shape) == 2:
            prediction_mask = np.expand_dims(prediction_mask, axis=0)
            is_multiclass = False
        else:
            is_multiclass = True
            
        for channel_idx, class_mask in enumerate(prediction_mask):
            # If multi-class, index 0 is usually the background, so we skip it.
            if is_multiclass and channel_idx == 0:
                continue
                
            # Determine label
            actual_class_id = channel_idx if is_multiclass else 1
            feature_type = class_names.get(actual_class_id, f"class_{actual_class_id}")
            
            # 1. Binarize
            binary_mask = (class_mask > self.threshold).astype(np.uint8)
            
            if np.max(binary_mask) == 0: # Skip if no predictions
                continue
            
            # 2. Clean noise (remove small regions)
            cleaned = clean_mask(binary_mask, min_area=self.min_area)
            
            # 3. Convert to GeoJSON Features
            geojson = mask_to_geojson(
                cleaned,
                min_area=self.min_area,
                simplify_tolerance=self.simplify_tolerance,
                feature_type=feature_type,
                extra_properties=extra_properties
            )
            all_features.extend(geojson["features"])
            
        return {
            "type": "FeatureCollection",
            "features": all_features
        }
