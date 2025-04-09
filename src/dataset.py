import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.utils import Sequence  # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore

from collections import Counter

IMG_SIZE = 224
DATA_DIR = "../data/UTKFace"
CACHE_PATH = "../data/utkface_data.npz"

def extractLabels(filename):
    try:
        age, gender, race = filename.split("_")[:3]
        return int(age), int(gender), int(race)
    except:
        return None

def ageToBin(age):
    return min(age // 10, 9)

def getImageFilepathsAndLabels():
    filepaths = []
    age_bins = []
    genders = []
    races = []

    for fname in os.listdir(DATA_DIR):
        label = extractLabels(fname)
        if label:
            age, gender, race = label
            img_path = os.path.join(DATA_DIR, fname)
            if not os.path.isfile(img_path):
                continue
            filepaths.append(img_path)
            age_bins.append(ageToBin(age))
            genders.append(gender)
            races.append(race)

    ageCount = Counter(age_bins)
    genderCount = Counter(genders)
    raceCount = Counter(races)

    print("age count:" + str(ageCount))
    print("gender count:" + str(genderCount))
    print("race count:" + str(raceCount))

    return filepaths, np.array(age_bins), np.array(genders), np.array(races)

def loadOrProcessData():
    if os.path.exists(CACHE_PATH):
        print("Loading cached data...")
        data = np.load(CACHE_PATH, allow_pickle=True)
        filepaths = data["filepaths"].tolist()
        age_bins = data["age_bins"]
        genders = data["genders"]
        races = data["races"]
    else:
        print("Processing dataset...")
        filepaths, age_bins, genders, races = getImageFilepathsAndLabels()
        np.savez(CACHE_PATH,
                filepaths=np.array(filepaths),
                age_bins=age_bins,
                genders=genders,
                races=races)
    return filepaths, age_bins, genders, races

class UTKFaceSequence(Sequence):
    def __init__(self, filepaths, age_labels, gender_labels, race_labels, batch_size=32, shuffle=True):
        self.filepaths = filepaths
        self.age_labels = age_labels
        self.gender_labels = gender_labels
        self.race_labels = race_labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.onEpochEnd()

        # Infer number of classes
        self.num_age_bins = 10
        self.num_gender_classes = 2
        self.num_race_classes = 5

    def __len__(self):
        return int(np.ceil(len(self.filepaths) / self.batch_size))

    def __getitem__(self, index):
        batch_paths = self.filepaths[index * self.batch_size:(index + 1) * self.batch_size]
        batch_ages = self.age_labels[index * self.batch_size:(index + 1) * self.batch_size]
        batch_genders = self.gender_labels[index * self.batch_size:(index + 1) * self.batch_size]
        batch_races = self.race_labels[index * self.batch_size:(index + 1) * self.batch_size]

        batch_images = []
        for path in batch_paths:
            img = cv2.imread(path)
            if img is not None:
                try:
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    img = img / 255.0
                except:
                    img = np.zeros((IMG_SIZE, IMG_SIZE, 3))
            else:
                img = np.zeros((IMG_SIZE, IMG_SIZE, 3))
            batch_images.append(img)

        X = np.array(batch_images, dtype=np.float32)

        # Ensure age labels are within the valid range (0 to 9)
        batch_ages = np.clip(batch_ages, 0, self.num_age_bins - 1)

        # One-hot encode the labels
        y_age = to_categorical(batch_ages, num_classes=self.num_age_bins)
        y_gender = to_categorical(batch_genders, num_classes=self.num_gender_classes)
        y_race = to_categorical(batch_races, num_classes=self.num_race_classes)

        return X, {
            'age_output': y_age,
            'gender_output': y_gender,
            'race_output': y_race
    }


    def onEpochEnd(self):
        if self.shuffle:
            indices = np.arange(len(self.filepaths))
            np.random.shuffle(indices)
            self.filepaths = [self.filepaths[i] for i in indices]
            self.age_labels = self.age_labels[indices]
            self.gender_labels = self.gender_labels[indices]
            self.race_labels = self.race_labels[indices]

def getGenerators(batch_size=32, test_size=0.2, shuffle=True):
    filepaths, age_bins, genders, races = loadOrProcessData()

    train_paths, test_paths, y_age_train, y_age_test, y_gender_train, y_gender_test, y_race_train, y_race_test = train_test_split(
        filepaths, age_bins, genders, races, test_size=test_size, random_state=42
    )

    train_gen = UTKFaceSequence(train_paths, y_age_train, y_gender_train, y_race_train, batch_size=batch_size, shuffle=shuffle)
    test_gen = UTKFaceSequence(test_paths, y_age_test, y_gender_test, y_race_test, batch_size=batch_size, shuffle=False)

    return train_gen, test_gen
