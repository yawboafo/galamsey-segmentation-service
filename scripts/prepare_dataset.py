import os
import shutil
import random
from tqdm import tqdm
import argparse

def prepare_dataset(raw_dir, processed_dir, train_ratio=0.8, val_ratio=0.1):
    """
    Organizes raw images and masks into train/val/test splits.
    Assumes raw_dir contains 'images' and 'masks' subfolders.
    """
    images_raw = os.path.join(raw_dir, "images")
    masks_raw = os.path.join(raw_dir, "masks")
    
    if not os.path.exists(images_raw) or not os.path.exists(masks_raw):
        print(f"Error: Could not find images/ and masks/ in {raw_dir}")
        return

    all_images = sorted([f for f in os.listdir(images_raw) if f.endswith(('.png', '.jpg', '.tif'))])
    random.shuffle(all_images)
    
    n = len(all_images)
    train_n = int(n * train_ratio)
    val_n = int(n * val_ratio)
    
    splits = {
        "train": all_images[:train_n],
        "val": all_images[train_n:train_n+val_n],
        "test": all_images[train_n+val_n:]
    }
    
    for split, files in splits.items():
        split_img_dir = os.path.join(processed_dir, split, "images")
        split_mask_dir = os.path.join(processed_dir, split, "masks")
        os.makedirs(split_img_dir, exist_ok=True)
        os.makedirs(split_mask_dir, exist_ok=True)
        
        print(f"Copying {len(files)} files to {split} split...")
        for f in tqdm(files):
            # Copy image
            shutil.copy2(os.path.join(images_raw, f), os.path.join(split_img_dir, f))
            # Copy mask (assuming same filename)
            mask_file = f # Adjust if mask extension/suffix differs
            shutil.copy2(os.path.join(masks_raw, mask_file), os.path.join(split_mask_dir, mask_file))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="./data/raw")
    parser.add_argument("--processed_dir", default="./data/processed")
    args = parser.parse_args()
    
    prepare_dataset(args.raw_dir, args.processed_dir)
