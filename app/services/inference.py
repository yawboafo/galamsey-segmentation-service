import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import numpy as np
import io
import requests
from app.config import settings
from typing import Optional, Union

class GalamseyInferenceService:
    def __init__(self, model_path: Optional[str] = None):
        # Determine model source
        checkpoint = model_path or settings.MODEL_PATH
        if not os.path.exists(checkpoint):
            print(f"Local model not found at {checkpoint}, falling back to {settings.MODEL_CHECKPOINT}")
            checkpoint = settings.MODEL_CHECKPOINT
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model from {checkpoint} on {self.device}")
        
        self.processor = SegformerImageProcessor.from_pretrained(checkpoint)
        self.model = SegformerForSemanticSegmentation.from_pretrained(checkpoint)
        self.model.to(self.device)
        self.model.eval()

    def load_image(self, image_source: Union[bytes, str]) -> Image.Image:
        """Loads PIL image from bytes or URL."""
        if isinstance(image_source, str) and image_source.startswith("http"):
            response = requests.get(image_source)
            image_data = response.content
        elif isinstance(image_source, bytes):
            image_data = image_source
        else:
            raise ValueError("Invalid image source. Must be bytes or a valid URL.")
        
        # Try rasterio for multispectral/SAR if needed in future
        # For now, we fallback to PIL which reads standard RGB formats
        try:
            image = Image.open(io.BytesIO(image_data))
            # Just to ensure we're not crashing on 4-band if our model isn't ready
            if image.mode != "RGB":
                image = image.convert("RGB")
            return image
        except Exception as e:
            raise ValueError(f"Failed to load image: {e}")

    def calculate_ndvi(self, image: Image.Image) -> float:
        """
        Mock NDVI calculation. 
        In actual production with Multispectral TIFFs, this uses (NIR - R) / (NIR + R).
        Here we return a simulated overall vegetation stress score for demo purposes.
        """
        import random
        # Normally: extract band 4 (NIR) and band 3 (Red)
        return random.uniform(-1.0, 1.0)

    @torch.no_grad()
    def predict(self, image: Image.Image, return_all_classes: bool = True):
        """
        Runs inference on the image. Retuns probability mask for multiple classes.
        """
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        logits = outputs.logits 
        
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=image.size[::-1], # H, W
            mode="bilinear",
            align_corners=False,
        )
        probs = torch.softmax(upsampled_logits, dim=1)
        
        if return_all_classes and probs.shape[1] > 2:
            # Return all class channels [num_classes, H, W]
            return probs[0].cpu().numpy()
        else:
            # Fallback for binary model (just 'galamsey')
            return probs[0, 1].cpu().numpy()

    def calculate_change(self, mask_t1: np.ndarray, mask_t2: np.ndarray) -> np.ndarray:
        """
        Computes the delta between two prediction masks.
        Positive values indicate new developments (e.g., new galamsey pits).
        """
        # Simple logical difference for binary arrays: new = t2 AND NOT t1
        delta = np.maximum(0, mask_t2 - mask_t1)
        return delta

import os
# Singleton instance
_service = None

def get_inference_service():
    global _service
    if _service is None:
        _service = GalamseyInferenceService()
    return _service
