import os
import numpy as np
#import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.utils import Sequence

IMG_SIZE = 224
DATA_DIR = "./data/UTKFace"
CACHE_PATH = "./data/utkface_data.npz"

def extract_labels(filename):
    try:
        age, gender, race = filename.split("_")[:3]
        return int(age), int(gender), int(race)
    except:
        return None

def age_to_bin(age):
    if age > 100:
        return 11
    return age // 10

def get_image_filepaths_and_labels():
    filepaths = []
    age_bins = []
    genders = []
    races = []

    for fname in os.listdir(DATA_DIR):
        label = extract_labels(fname)
        if label:
            age, gender, race = label
            img_path = os.path.join(DATA_DIR, fname)
            if not os.path.isfile(img_path):
                continue
            filepaths.append(img_path)
            age_bins.append(age_to_bin(age))
            genders.append(gender)
            races.append(race)

    return filepaths, np.array(age_bins), np.array(genders), np.array(races)

def load_or_process_data():
    if os.path.exists(CACHE_PATH):
        print("Loading cached data...")
        data = np.load(CACHE_PATH, allow_pickle=True)
        filepaths = data["filepaths"].tolist()
        age_bins = data["age_bins"]
        genders = data["genders"]
        races = data["races"]
    else:
        print("Processing dataset...")
        filepaths, age_bins, genders, races = get_image_filepaths_and_labels()
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
        self.on_epoch_end()

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
                    batch_images.append(img)
                except:
                    batch_images.append(np.zeros((IMG_SIZE, IMG_SIZE, 3)))  # fallback image
            else:
                batch_images.append(np.zeros((IMG_SIZE, IMG_SIZE, 3)))  # fallback if unreadable

        X = np.array(batch_images, dtype=np.float32)
        return X, {
            'age': batch_ages,
            'gender': batch_genders,
            'race': batch_races
        }

    def on_epoch_end(self):
        if self.shuffle:
            indices = np.arange(len(self.filepaths))
            np.random.shuffle(indices)
            self.filepaths = [self.filepaths[i] for i in indices]
            self.age_labels = self.age_labels[indices]
            self.gender_labels = self.gender_labels[indices]
            self.race_labels = self.race_labels[indices]

def get_generators(batch_size=32, test_size=0.2, shuffle=True):
    filepaths, age_bins, genders, races = load_or_process_data()

    train_paths, test_paths, y_age_train, y_age_test, y_gender_train, y_gender_test, y_race_train, y_race_test = train_test_split(
        filepaths, age_bins, genders, races, test_size=test_size, random_state=42
    )

    train_gen = UTKFaceSequence(train_paths, y_age_train, y_gender_train, y_race_train, batch_size=batch_size, shuffle=shuffle)
    test_gen = UTKFaceSequence(test_paths, y_age_test, y_gender_test, y_race_test, batch_size=batch_size, shuffle=False)

    return train_gen, test_gen
