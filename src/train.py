from utils import singleTestModel, UTKLabelType, removeCorruptImages
from dataset import getGenerators
from model import buildModel
from keras.models import load_model # type: ignore
import os

if __name__ == "__main__":
    data_location = "../data/UTKFace"
    removeCorruptImages(data_location)
    # Choose the label type you want to train on
    labelType = UTKLabelType.RACE  # Change to AGE or RACE as needed

    # Get generators for training and testing
    train_gen, test_gen = getGenerators(labelType=labelType, batch_size=32)

    # Build and compile the model for the chosen label type
    model = buildModel(labelType)
    
    # Train the model
    model.fit(
        train_gen,
        validation_data=test_gen,
        epochs=10,
        verbose=1
    )

    model.save(f"../models/{labelType}.keras")