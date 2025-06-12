import cv2
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
import os
import numpy as np
from util import UTKLabelType
from collections import Counter
import csv
from collections import Counter


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

DATA_DIR = "archive/UTK faces/Dataset/"
AGE_BINS = 9
IMG_SIZE = 224
def extractLabels(filename):
    try:
        age, gender, race = filename.split("_")[:3]
        return int(age), int(gender), int(race)
    except:
        return None

def ageToBin(age):
    return min(age // 1, 7)



def extractCSVLabels(csvFilePath, dataLabel: UTKLabelType):
    with open(csvFilePath, newline="") as file:
        reader = csv.reader(file)
        next(reader)
        columns = list(zip(*reader))
        paths = [path.replace("\\", "/") for path in columns[0]]
        if dataLabel == UTKLabelType.AGE:
            label = 1
        elif dataLabel == UTKLabelType.GENDER:
            label = 2
        else:
            label = 3

        labels = list(columns[label])

        valid_paths = []
        valid_labels = []

        for path, label in zip(paths, labels):
            # print(label)
            full_path = os.path.join(DATA_DIR, os.path.basename(path))  # ensure full path
            img = cv2.imread(full_path)
            if img is not None:
                valid_paths.append(full_path)
                valid_labels.append(int(label))  # ensure labels are int
            else:
                print(f"Skipping unreadable file: {full_path}")

        return valid_paths, valid_labels



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

CACHE_PATH_TEMPLATE = "../archive/UTK faces/{}.npz"
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

class UTKFaceSequence(Dataset):
    def __init__( self ,labeltype , filepaths , data  ):
        self.labelType = labeltype  
        self.filepaths = filepaths
        self.data = data
        self.transform = transform
        # Infer number of classes
        if labeltype == UTKLabelType.AGE:
            self.num_of_classes = 8
        elif labeltype == UTKLabelType.GENDER:
            self.num_of_classes = 2
        elif labeltype == UTKLabelType.RACE: 
            self.num_of_classes = 5
        else:
            raise Exception("Wrong label")

        
    def __len__(self):
        return (len(self.filepaths))

    def __getitem__(self, index):
        img = Image.open(self.filepaths[index]).convert('RGB')
        img = self.transform(img)

        label = int(self.data[index])
        return img, label
    

def getGenerators(filepaths: list, label_data:list, labelType: UTKLabelType,  test_size=0.2):
    # filepaths, label_data = loadOrProcessData(labelType)
    label_data = list(label_data)  # Ensure it's a list, not NumPy
    # stratify = label_data if len(set(label_data)) <= 10 else None

    train_paths, test_paths, y_train, y_test = train_test_split(
        filepaths, label_data, test_size=test_size, random_state=42, shuffle=True , stratify=label_data
    )
    print("Train class dist:", Counter(y_train))
    print("Test class dist:", Counter(y_test))
    
    train_gen = UTKFaceSequence(labelType, train_paths, y_train)
    test_gen = UTKFaceSequence(labelType, test_paths, y_test)

    trainloader = DataLoader ( train_gen , batch_size = 32 , shuffle = True)
    testloader = DataLoader ( test_gen , batch_size = 32 , shuffle = False)

    return trainloader , testloader