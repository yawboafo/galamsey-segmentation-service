import os
from PIL import Image
import numpy as np
import argparse
from tqdm import tqdm

def validate_dataset(dataset_dir):
    """
    Validates that images and masks match in dimensions and that masks are binary.
    """
    subdirs = ["train", "val", "test"]
    
    for split in subdirs:
        img_dir = os.path.join(dataset_dir, split, "images")
        mask_dir = os.path.join(dataset_dir, split, "masks")
        
        if not os.path.exists(img_dir):
            continue
            
        print(f"Validating {split} split...")
        images = sorted(os.listdir(img_dir))
        
        for img_name in tqdm(images):
            img_path = os.path.join(img_dir, img_name)
            mask_path = os.path.join(mask_dir, img_name)
            
            if not os.path.exists(mask_path):
                print(f"Error: Mask missing for {img_name}")
                continue
                
            img = Image.open(img_path)
            mask = Image.open(mask_path)
            
            # Check dimensions
            if img.size != mask.size:
                print(f"Error: Dimension mismatch for {img_name} {img.size} vs {mask.size}")
                
            # Check mask values
            mask_arr = np.array(mask)
            unique_values = np.unique(mask_arr)
            # Standard binary mask should have 0 and (1 or 255)
            # If it has more than 2-3 values, it might be multiclass or noisy
            if len(unique_values) > 2:
                print(f"Warning: Mask {img_name} has multiple values: {unique_values}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="./data/processed")
    args = parser.parse_args()
    validate_dataset(args.dataset_dir)
