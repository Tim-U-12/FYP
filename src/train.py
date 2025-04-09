from utils import singleTestModel
from dataset import getGenerators
from model import buildModel


if __name__ == "__main__":
    train_gen, test_gen = getGenerators(batch_size=32)

    model = buildModel()
    model.summary()
    
    model.fit(
        train_gen,
        validation_data=test_gen,
        epochs=10,
        verbose=1
    )

    
    predictIMG = singleTestModel("../test-face.jpg")
    