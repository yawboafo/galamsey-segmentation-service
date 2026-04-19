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

    def process(self, prediction_mask: np.ndarray) -> Dict[str, Any]:
        """
        Takes raw probability mask and converts to cleaned GeoJSON.
        """
        # 1. Binarize
        binary_mask = (prediction_mask > self.threshold).astype(np.uint8)
        
        # 2. Clean noise (remove small regions)
        cleaned = clean_mask(binary_mask, min_area=self.min_area)
        
        # 3. Convert to GeoJSON Features
        geojson = mask_to_geojson(
            cleaned,
            min_area=self.min_area,
            simplify_tolerance=self.simplify_tolerance
        )
        
        return geojson
