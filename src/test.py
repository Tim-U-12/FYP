import os
from utils import UTKLabelType, removeCorruptImages, singleTestModel
from keras.models import load_model # type: ignore

if __name__ == "__main__":
    data_location = "data/UTKFace"
    removeCorruptImages(data_location)
    # Choose the label type you want to train on
    labelType = UTKLabelType.RACE  # Change to AGE or RACE as needed
    script_dir = os.path.dirname(os.path.abspath(__file__))  # /Users/timu/Repos/FYP/src
    model_path = os.path.join(script_dir, "..", "models", f"{labelType}.keras")
    model_path = os.path.abspath(model_path)  # fully resolved absolute path

    print(f"Resolved model path: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file does not exist: {model_path}")

    model = load_model(model_path)

    predictIMG = singleTestModel(os.path.join(script_dir, "..", "test-face.jpg"), model=model, labelType=labelType)

    print(predictIMG)