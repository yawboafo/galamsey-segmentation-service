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
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
        elif isinstance(image_source, bytes):
            image = Image.open(io.BytesIO(image_source)).convert("RGB")
        else:
            raise ValueError("Invalid image source. Must be bytes or a valid URL.")
        return image

    @torch.no_grad()
    def predict(self, image: Image.Image):
        """
        Runs inference on the image and returns probability mask for class 1.
        """
        # Preprocess
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        # Inference
        outputs = self.model(**inputs)
        logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)
        
        # Upsample logits to original image size
        # Segformer output is 1/4 of input size
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=image.size[::-1], # H, W
            mode="bilinear",
            align_corners=False,
        )
        
        # Get softmax probabilities
        probs = torch.softmax(upsampled_logits, dim=1)
        
        # We only care about the 'galamsey' class (index 1)
        # Result shape: [H, W]
        prediction = probs[0, 1].cpu().numpy()
        
        return prediction

import os
# Singleton instance
_service = None

def get_inference_service():
    global _service
    if _service is None:
        _service = GalamseyInferenceService()
    return _service
