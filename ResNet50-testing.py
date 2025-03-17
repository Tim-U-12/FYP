import tensorflow as tf
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

model = ResNet50(weights="imagenet")

def load_image(image_path):
    img = Image.open(image_path).resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

image_path = "test-face.jpg"
image = load_image(image_path)

predictions = model.predict(image)
decoded_predictions = decode_predictions(predictions, top=3)[0]

for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
    print(f"{i+1}: {label} ({score:.2f})")