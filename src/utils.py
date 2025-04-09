from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras.applications.resnet50 import preprocess_input # type: ignore
import numpy as np

from dataset import loadOrProcessData

def singleTestModel(img_path):
    # Load the image
    img = image.load_img(img_path, target_size=(224, 224))

    # Convert to array and preprocess
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # make it a batch of 1
    img_array = preprocess_input(img_array)  # apply ResNet50 preprocessing
    return img_array

if __name__ == "__main__":
    IMG_SIZE = 224
    DATA_DIR = "../data/UTKFace"
    CACHE_PATH = "../data/utkface_data.npz"
    loadOrProcessData()