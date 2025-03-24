import os
import cv2

IMG_SIZE = 224
DATA_DIR = "data/UTKFace"

def extract_labels(filename):
    try:
        age, gender, race = filename.split("_")[:3]
        return int(age), int(gender), int(race)
    except:
        return None

if __name__ == "__main__":
    for fname in os.listdir(DATA_DIR):
        print(extract_labels(fname))