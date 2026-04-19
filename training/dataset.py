import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import List, Tuple, Optional

class GalamseyDataset(Dataset):
    """
    Custom Dataset for Galamsey Segmentation.
    Expects images and masks in matching subdirectories.
    """
    def __init__(
        self, 
        image_dir: str, 
        mask_dir: str, 
        processor,
        image_size: int = 512,
        transform: Optional[A.Compose] = None
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.processor = processor
        self.image_size = image_size
        self.transform = transform
        
        # List all image files
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.tif'))])
        
        # Basic validation: ensure masks exist for all images
        for img_name in self.images:
            mask_path = os.path.join(mask_dir, img_name)
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask not found for image: {img_name} at {mask_path}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)
        
        # Load image and mask
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
        
        # Normalize mask to binary (0, 1)
        # Assuming 1 or 255 represents galamsey
        mask = (mask > 0).astype(np.uint8)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        
        # SegFormer expectations: 
        # Inputs: [batch_size, 3, height, width]
        # Labels: [batch_size, height, width]
        
        # Use SegformerImageProcessor for necessary preprocessing (scaling, normalization)
        encoded_inputs = self.processor(image, return_tensors="pt")
        
        # Remove batch dimension added by processor
        inputs = {k: v.squeeze(0) for k, v in encoded_inputs.items()}
        inputs["labels"] = torch.from_numpy(mask).long()
        
        return inputs

def get_train_transform(image_size: int = 512):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.2),
        A.OneOf([
            A.ElasticTransform(p=0.3),
            A.GridDistortion(p=0.1),
            A.OpticalDistortion(p=0.3),
        ], p=0.2),
        A.OneOf([
            A.GaussNoise(p=0.2),
            A.RandomBrightnessContrast(p=0.3),
        ], p=0.2),
    ])

def get_val_transform(image_size: int = 512):
    return A.Compose([
        A.Resize(image_size, image_size),
    ])
