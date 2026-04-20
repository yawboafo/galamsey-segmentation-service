import os
import numpy as np
from PIL import Image, ImageDraw
import random

def generate_mock_data(output_dir="./data/raw", num_samples=20, image_size=(512, 512)):
    """
    Generates synthetic multi-class satellite image chips and integer masks.
    Classes: 0 (Bg), 1 (Galamsey), 2 (Veg Loss), 3 (Road), 4 (Water)
    """
    img_dir = os.path.join(output_dir, "images")
    mask_dir = os.path.join(output_dir, "masks")
    
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    
    print(f"Generating {num_samples} mock chips in {output_dir}...")
    
    for i in range(num_samples):
        # 1. Base RGB
        img_array = np.random.randint(40, 140, (image_size[0], image_size[1], 3), dtype=np.uint8)
        img_array[:, :, 1] = np.clip(img_array[:, :, 1] + 30, 0, 255) # Greenish baseline
        
        # 2. Mask
        mask = Image.new("L", image_size, 0)
        draw = ImageDraw.Draw(mask)
        
        # Water (Class 4) - randomly draw a river across
        if random.random() > 0.5:
            y1, y2 = random.randint(0, 512), random.randint(0, 512)
            draw.line([(0, y1), (512, y2)], fill=4, width=random.randint(15, 40))
            
        # Road (Class 3) - brown pathways
        for _ in range(random.randint(0, 2)):
            x1, y1 = random.randint(0, 512), random.randint(0, 512)
            x2, y2 = random.randint(0, 512), random.randint(0, 512)
            draw.line([(x1, y1), (x2, y2)], fill=3, width=random.randint(4, 10))

        # Veg Loss (Class 2) - large sparse barren zones
        for _ in range(random.randint(1, 3)):
            x, y = random.randint(50, 460), random.randint(50, 460)
            r = random.randint(40, 80)
            draw.ellipse([x-r, y-r, x+r, y+r], fill=2)

        # Galamsey Pit (Class 1) - inner dense targets
        for _ in range(random.randint(1, 4)):
            x, y = random.randint(100, 400), random.randint(100, 400)
            r = random.randint(15, 30)
            draw.ellipse([x-r, y-r, x+r, y+r], fill=1)
            
        filename = f"mock_chip_{i:03d}.png"
        Image.fromarray(img_array).save(os.path.join(img_dir, filename))
        mask.save(os.path.join(mask_dir, filename))

    print("Mock multi-class data generation complete!")

if __name__ == "__main__":
    generate_mock_data()
