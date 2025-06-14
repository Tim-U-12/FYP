from utils import UTKLabelType, removeCorruptImages, evaluateModel
from dataset import ageToBin, extractCSVLabels, getGenerators, loadOrProcessData
from model import buildModel
from keras.models import load_model # type: ignore
import os

if __name__ == "__main__":
    data_location = "data/UTKFace"
    # removeCorruptImages(data_location)
    labelType = UTKLabelType.AGE
    
    ###################################################################
    # Unbalanced 
    ###################################################################
    
    # filepaths, label_data = loadOrProcessData(labelType)
    # train_gen, test_gen = getGenerators(filepaths, label_data, labelType=labelType, batch_size=32)
    # model = buildModel(labelType)
    
    # model.fit(
    #     train_gen,
    #     validation_data=test_gen,
    #     epochs=10,
    #     verbose=1
    # )
    
    # model.save(f"../models/{labelType}_unbal.keras")
    
    ###################################################################
    # Balanced 
    ###################################################################

    csvPath = f"./data/BalancedDatasets/balanced{labelType.name}.csv"
    filepaths, label_data = extractCSVLabels(csvPath,labelType)

    train_gen, test_gen = getGenerators(filepaths, label_data, labelType=labelType, batch_size=32)
    model = buildModel(labelType)
    
    model.fit(
        train_gen,
        validation_data=test_gen,
        epochs=10,
        verbose=1
    )
    
    model.save(f"./models/{labelType}_bal.keras")

    ###################################################################
    # Evaluate the model 
    ###################################################################
    
    evaluateModel(model, test_gen, labelType)