from PIL import Image
import os

def single_image_undersample(input_path, output_path, quality=30):
    """Reduce image quality/size (simulate undersampling)"""
    # Downscale image
    img = Image.open(input_path)
    img = img.resize((img.width//2, img.height//2))  # Halve resolution
    
    # Save with lower quality
    img.save(output_path, quality=quality)
    print(f"Undersampled image saved to {output_path}")

# Usage
single_image_undersample("test-face.jpg", "test-face_undersampled.jpg")