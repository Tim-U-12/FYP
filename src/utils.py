from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras.applications.resnet50 import preprocess_input # type: ignore
from enum import Enum, auto
import numpy as np

class UTKLabelType(Enum):
    AGE = auto()
    GENDER = auto()
    RACE = auto()

def singleTestModel(img_path):
    # Load the image
    img = image.load_img(img_path, target_size=(224, 224))

    # Convert to array and preprocess
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # make it a batch of 1
    img_array = preprocess_input(img_array)  # apply ResNet50 preprocessing
    return img_array

