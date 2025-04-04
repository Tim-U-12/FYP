from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np

def singleTestModel():
    # Load the image
    img_path = "../test-face.jpg"  # adjust to your actual image path
    img = image.load_img(img_path, target_size=(224, 224))

    # Convert to array and preprocess
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # make it a batch of 1
    img_array = preprocess_input(img_array)  # apply ResNet50 preprocessing
    return img_array