from torchvision import datasets, transforms
from enum import Enum, auto
import numpy as np
from PIL import Image
import os

class UTKLabelType(Enum):
    AGE = auto()
    GENDER = auto()
    RACE = auto()

    def toString(self):
        return self.name 


def removeCorruptImages():
    for fname in os.listdir("../archive/UTK Faces/dataset"):
        fpath = os.path.join("../archive/UTK Faces/dataset", fname)
        try:
            img = Image.open(fpath)
            img.verify()
        except Exception as e:
            print(f"Corrupt: {fpath}, error: {e}")
            os.remove(fpath) 