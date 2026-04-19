import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import numpy as np
import io
from typing import Dict, Any, List
import cv2
from shapely.geometry import Polygon, mapping

class EndpointHandler:
    def __init__(self, path=""):
        # Load model and processor from the path
        self.processor = SegformerImageProcessor.from_pretrained(path)
        self.model = SegformerForSemanticSegmentation.from_pretrained(path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Args:
            data (:obj:`dict`):
                subset of parameters:
                    - "inputs": (bytes) raw image data
                    - "threshold": (float) binarization threshold
                    - "min_area": (int) minimum pixel area
        Return:
            A :obj:`dict` with GeoJSON and confidence
        """
        # 1. Parse Input
        inputs = data.pop("inputs", data)
        threshold = data.pop("threshold", 0.5)
        min_area = data.pop("min_area", 50)
        simplify_tolerance = data.pop("simplify_tolerance", 0.1)

        if isinstance(inputs, bytes):
            image = Image.open(io.BytesIO(inputs)).convert("RGB")
        else:
            # Handle base64 or other formats if needed
            image = inputs

        # 2. Inference
        with torch.no_grad():
            encoding = self.processor(images=image, return_tensors="pt").to(self.device)
            outputs = self.model(**encoding)
            logits = outputs.logits
            
            # Upsample
            upsampled_logits = torch.nn.functional.interpolate(
                logits,
                size=image.size[::-1], # H, W
                mode="bilinear",
                align_corners=False,
            )
            probs = torch.softmax(upsampled_logits, dim=1)
            prediction_mask = probs[0, 1].cpu().numpy()

        # 3. Post-Processing (Simplified version of our app logic)
        binary_mask = (prediction_mask > threshold).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        features = []
        for contour in contours:
            pts = contour.reshape(-1, 2)
            if len(pts) < 3: continue
            
            poly = Polygon(pts)
            if poly.area < min_area: continue
            
            if simplify_tolerance > 0:
                poly = poly.simplify(simplify_tolerance, preserve_topology=True)
                
            features.append({
                "type": "Feature",
                "geometry": mapping(poly),
                "properties": {"area_px": float(poly.area)}
            })

        return {
            "success": True,
            "confidence": float(np.max(prediction_mask)),
            "area_geojson": {
                "type": "FeatureCollection",
                "features": features
            }
        }
