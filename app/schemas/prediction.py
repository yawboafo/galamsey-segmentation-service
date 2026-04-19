from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class PredictionRequest(BaseModel):
    image_url: Optional[str] = Field(None, description="URL of the satellite image chip")
    threshold: Optional[float] = Field(None, description="Confidence threshold for mask binarization")
    min_area: Optional[int] = Field(None, description="Minimum pixel area for detection")
    simplify_tolerance: Optional[float] = Field(None, description="Polygon simplification tolerance")

class FeatureGeometry(BaseModel):
    type: str
    coordinates: List[Any]

class Feature(BaseModel):
    type: str = "Feature"
    geometry: Dict[str, Any]
    properties: Dict[str, Any]

class GeoJSONResponse(BaseModel):
    type: str = "FeatureCollection"
    features: List[Feature]

class PredictionResponse(BaseModel):
    success: bool
    confidence: float
    area_geojson: GeoJSONResponse
    metadata: Dict[str, Any]
