from utils import singleTestModel, UTKLabelType, removeCorruptImages
from dataset import getGenerators
from model import buildModel
from keras.models import load_model 
import os

if __name__ == "__main__":
    # data_location = "../data/UTKFace"
    # removeCorruptImages(data_location)
    # # Choose the label type you want to train on
    labelType = UTKLabelType.RACE  # Change to AGE or RACE as needed

    # # Get generators for training and testing
    # train_gen, test_gen = getGenerators(labelType=labelType, batch_size=32)

    # # Build and compile the model for the chosen label type
    # model = buildModel(labelType)
    
    # # Train the model
    # model.fit(
    #     train_gen,
    #     validation_data=test_gen,
    #     epochs=10,
    #     verbose=1
    # )

    # Single image prediction

    script_dir = os.path.dirname(os.path.abspath(__file__))  # /Users/timu/Repos/FYP/src
    model_path = os.path.join(script_dir, "..", "models", f"{labelType}.keras")
    model_path = os.path.abspath(model_path)  # fully resolved absolute path

    print(f"Resolved model path: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file does not exist: {model_path}")

    model = load_model(model_path)

    predictIMG = singleTestModel(os.path.join(script_dir, "..", "test-face.jpg"), model=model, labelType=labelType)

    print(predictIMG)

    # model.save(f"../models/{labelType}.keras")