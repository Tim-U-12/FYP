from utils import singleTestModel, UTKLabelType, removeCorruptImages
from dataset import getGenerators
from model import buildModel

if __name__ == "__main__":
    removeCorruptImages
    # Choose the label type you want to train on
    labelType = UTKLabelType.GENDER  # Change to AGE or RACE as needed

    # Get generators for training and testing
    train_gen, test_gen = getGenerators(labelType=labelType, batch_size=32)

    # Build and compile the model for the chosen label type
    model = buildModel(labelType)
    model.summary()
    
    # Train the model
    model.fit(
        train_gen,
        validation_data=test_gen,
        epochs=10,
        verbose=1
    )

    # Single image prediction
    predictIMG = singleTestModel("../test-face.jpg", model=model, labelType=labelType)
