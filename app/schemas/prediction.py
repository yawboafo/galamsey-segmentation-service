from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum

class AnalysisMode(str, Enum):
    rgb = "rgb"
    multispectral = "multispectral"
    sar = "sar"

class PredictionRequest(BaseModel):
    image_url: Optional[str] = Field(None, description="URL of the satellite image chip")
    analysis_mode: AnalysisMode = Field(AnalysisMode.rgb, description="Mode of imaging data")
    threshold: Optional[float] = Field(None, description="Confidence threshold for mask binarization")
    min_area: Optional[int] = Field(None, description="Minimum pixel area for detection")
    simplify_tolerance: Optional[float] = Field(None, description="Polygon simplification tolerance")

class TimeSeriesRequest(BaseModel):
    image_url_t1: str = Field(..., description="Baseline image URL (t1)")
    image_url_t2: str = Field(..., description="Follow-up image URL (t2)")
    analysis_mode: AnalysisMode = Field(AnalysisMode.rgb, description="Mode of imaging data")
    threshold: Optional[float] = Field(None, description="Confidence threshold")
    min_area: Optional[int] = Field(None, description="Minimum pixel area")
    simplify_tolerance: Optional[float] = Field(None, description="Simplify tolerance")

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
