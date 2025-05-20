import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image

def single_image_oversample(input_path, output_dir, num_copies=5):
    """Generate augmented variations (simulate oversampling)"""
    os.makedirs(output_dir, exist_ok=True)
    img = Image.open(input_path)
    img_array = np.expand_dims(np.array(img), axis=0)
    
    # Configure augmentations
    datagen = ImageDataGenerator(
        rotation_range=25,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )
    
    # Generate copies
    prefix = os.path.splitext(os.path.basename(input_path))[0]
    datagen.flow(
        img_array,
        batch_size=1,
        save_to_dir=output_dir,
        save_prefix=f'aug_{prefix}',
        save_format='jpg'
    ).next()  # Generates num_copies images
    print(f"Generated {num_copies} variations in {output_dir}")

# Usage
single_image_oversample("test-face.jpg", "augmented_faces", num_copies=5)