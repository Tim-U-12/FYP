from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras.applications.resnet50 import preprocess_input # type: ignore
from enum import Enum, auto
import numpy as np
import os
from PIL import Image


class UTKLabelType(Enum):
    AGE = auto()
    GENDER = auto()
    RACE = auto()

def singleTestModel(img_path, model, labelType: UTKLabelType):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Predict
    preds = model.predict(img_array)

    # Decode prediction based on label type
    pred_class = np.argmax(preds[0])  # output shape: (1, num_classes)

    if labelType == UTKLabelType.AGE:
        return f"Predicted Age Bin: {pred_class} (i.e., {pred_class * 10}-{pred_class * 10 + 9})"
    elif labelType == UTKLabelType.GENDER:
        return f"Predicted Gender: {'Male' if pred_class == 1 else 'Female'}"
    elif labelType == UTKLabelType.RACE:
        race_labels = ["White", "Black", "Asian", "Indian", "Other"]
        return f"Predicted Race: {race_labels[pred_class]}"
    else:
        return "Invalid label type"

def removeCorruptImages():
    for fname in os.listdir("../data/UTKFace"):
        fpath = os.path.join("../data/UTKFace", fname)
        try:
            img = Image.open(fpath)
            img.verify()
        except Exception as e:
            print(f"Corrupt: {fpath}, error: {e}")
            os.remove(fpath) 