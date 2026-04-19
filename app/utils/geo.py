import numpy as np
import cv2
from shapely.geometry import Polygon, MultiPolygon, mapping
import json
from typing import Dict, Any, List, Optional

def mask_to_geojson(
    mask: np.ndarray, 
    threshold: float = 0.5, 
    min_area: int = 50,
    simplify_tolerance: float = 0.1,
    transform = None
) -> Dict[str, Any]:
    """
    Converts a binary mask to a GeoJSON FeatureCollection of simplified polygons.
    
    Args:
        mask: Probability mask or binary mask (numpy array)
        threshold: Threshold for binarization if mask contains probabilities
        min_area: Minimum pixel area to keep a polygon
        simplify_tolerance: Tolerance for Douglas-Peucker simplification
        transform: Rasterio transform if geolocation is available
    """
    # Ensure binary mask
    if mask.dtype != np.uint8:
        binary_mask = (mask > threshold).astype(np.uint8)
    else:
        binary_mask = mask

    # Find contours using OpenCV
    contours, hierarchy = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    features = []
    
    for contour in contours:
        # Move contour to pixel coordinates
        # OpenCV returns (x, y) coordinates
        pts = contour.reshape(-1, 2)
        
        if len(pts) < 3:
            continue
            
        # Create Shapely polygon
        poly = Polygon(pts)
        
        # Check area
        if poly.area < min_area:
            continue
            
        # Simplify geometry
        if simplify_tolerance > 0:
            poly = poly.simplify(simplify_tolerance, preserve_topology=True)
        
        # Geotransform if transform is provided
        if transform:
            # TODO: Apply rasterio transform to points
            # For now, we remain in pixel coordinates
            pass
            
        # Add to features
        feature = {
            "type": "Feature",
            "geometry": mapping(poly),
            "properties": {
                "area_px": float(poly.area),
                "is_galamsey": True
            }
        }
        features.append(feature)

    return {
        "type": "FeatureCollection",
        "features": features
    }

def clean_mask(mask: np.ndarray, min_area: int = 50) -> np.ndarray:
    """
    Removes small disconnected regions from the mask.
    """
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    # stats[i, cv2.CC_STAT_AREA] is the area of the i-th component
    new_mask = np.zeros_like(mask)
    for i in range(1, nb_components):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            new_mask[output == i] = 1
    return new_mask
