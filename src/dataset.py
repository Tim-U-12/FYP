import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.utils import Sequence  # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.applications.resnet50 import preprocess_input # type:ignore
from utils import UTKLabelType

from collections import Counter

IMG_SIZE = 224
DATA_DIR = "../data/UTKFace"
AGE_BINS = 9


def extractLabels(filename):
    try:
        age, gender, race = filename.split("_")[:3]
        return int(age), int(gender), int(race)
    except:
        return None

def ageToBin(age):
    return min(age // 10, 9)

def getImageFilepathsAndLabels(dataLabel: UTKLabelType):
    filepaths = []
    labelValues = []
    
    for fname in os.listdir(DATA_DIR):
        label = extractLabels(fname)
        if label:
            age, gender, race = label
            img_path = os.path.join(DATA_DIR, fname)
            if not os.path.isfile(img_path):
                continue
            filepaths.append(img_path)
            
            if dataLabel == UTKLabelType.AGE:
                labelValues.append(ageToBin(age))
            elif dataLabel == UTKLabelType.GENDER:
                labelValues.append(gender)
            elif dataLabel == UTKLabelType.RACE:
                labelValues.append(race)
            else:
                raise Exception("Incorrect dataLabel")
    
    labelCount= Counter(labelValues)
    print("{}".format(labelCount))
    return filepaths, np.array(labelValues)

CACHE_PATH_TEMPLATE = "../data/{}.npz"
def loadOrProcessData(dataLabel: UTKLabelType):
    CACHE_PATH = CACHE_PATH_TEMPLATE.format(dataLabel)
    if os.path.exists(CACHE_PATH):
        print("Loading cached data...")
        data = np.load(CACHE_PATH, allow_pickle=True)
        filepaths = data["filepaths"].tolist()
        labelData = data["dataValues"]
    else:
        print("Processing dataset...")
        filepaths, dataValues = getImageFilepathsAndLabels(dataLabel)  # <-- fix is here
        np.savez(CACHE_PATH,
                filepaths=np.array(filepaths),
                dataValues=dataValues)
        labelData = dataValues  # assign the variable properly
    return filepaths, labelData


class UTKFaceSequence(Sequence):
    def __init__(self, labelType, filepaths, data, batch_size=32, shuffle=True):
        self.labelType = labelType
        self.filepaths = filepaths
        self.data = data 
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.onEpochEnd()

        # Infer number of classes
        if labelType == UTKLabelType.AGE:
            self.num_of_classes = 10
        elif labelType == UTKLabelType.GENDER:
            self.num_of_classes = 2
        elif labelType == UTKLabelType.RACE: 
            self.num_of_classes = 5
        else:
            raise Exception("Wrong label")

    def __len__(self):
        return int(np.ceil(len(self.filepaths) / self.batch_size))

    def __getitem__(self, index):
        batch_paths = self.filepaths[index * self.batch_size:(index + 1) * self.batch_size]
        batch_data = self.data[index * self.batch_size:(index + 1) * self.batch_size]

        batch_images = []
        for path in batch_paths:
            img = cv2.imread(path)
            if img is not None:
                try:
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    img = preprocess_input(img)   
                except:
                    img = np.zeros((IMG_SIZE, IMG_SIZE, 3))
            else:
                img = np.zeros((IMG_SIZE, IMG_SIZE, 3))
            batch_images.append(img)

        X = np.array(batch_images, dtype=np.float32)

        # One-hot encode the labels
        y_data = to_categorical(batch_data, self.num_of_classes)

        return X, y_data 

    def onEpochEnd(self):
        if self.shuffle:
            indices = np.arange(len(self.filepaths))
            np.random.shuffle(indices)
            self.filepaths = [self.filepaths[i] for i in indices]
            self.data = self.data[indices]

def getGenerators(labelType: UTKLabelType, batch_size=32, test_size=0.2, shuffle=True):
    filepaths, label_data = loadOrProcessData(labelType)

    train_paths, test_paths, y_train, y_test = train_test_split(
        filepaths, label_data, test_size=test_size, random_state=42, shuffle=shuffle
    )

    train_gen = UTKFaceSequence(labelType, train_paths, y_train, batch_size=batch_size, shuffle=shuffle)
    test_gen = UTKFaceSequence(labelType, test_paths, y_test, batch_size=batch_size, shuffle=False)

    return train_gen, test_gen