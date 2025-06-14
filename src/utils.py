from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras.applications.resnet50 import preprocess_input # type: ignore
from enum import Enum, auto
import numpy as np
from PIL import Image, UnidentifiedImageError
import os, sys, io
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    confusion_matrix, 
    classification_report, 
    roc_curve, 
    auc
)
import matplotlib.pyplot as plt

class UTKLabelType(Enum):
    AGE = auto()
    GENDER = auto()
    RACE = auto()

def singleTestModel(img_path, model, labelType: UTKLabelType):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Predict
    preds = model.predict(img_array)

    # Decode prediction based on label type
    pred_class = np.argmax(preds[0])  # output shape: (1, num_classes)

    if labelType == UTKLabelType.AGE:
        return f"Predicted Age Bin: {pred_class} (i.e., {pred_class * 10}-{pred_class * 10 + 9})"
    elif labelType == UTKLabelType.GENDER:
        return f"Predicted Gender: {'Male' if pred_class == 1 else 'Female'}"
    elif labelType == UTKLabelType.RACE:
        race_labels = ["White", "Black", "Asian", "Indian", "Other"]
        return f"Predicted Race: {race_labels[pred_class]}"
    else:
        return "Invalid label type"

def removeCorruptImages(folder):
    removed = 0
    for fname in os.listdir(folder):
        fpath = os.path.join(folder, fname)
        if not os.path.isfile(fpath):
            continue

        try:
            with Image.open(fpath) as img:
                img.verify()
        except (UnidentifiedImageError, OSError, IOError) as e:
            print(f"🗑️ Removing: {fpath} ← {e}")
            os.remove(fpath)
            removed += 1
    print(f"\nRemoved {removed} corrupt or unreadable images.")

def evaluateModel(model, test_gen, labelType):
    """
    Evaluates a Keras model using a test generator and prints metrics.

    Parameters:
    - model: Trained Keras model.
    - test_gen: Data generator that yields (X, y) batches.
    - labelType: Optional, to add context to report titles (e.g., AGE, GENDER, RACE).
    """
    y_true = []
    y_pred = []
    y_prob = []

    try:
        for batch_x, batch_y in test_gen:
            if len(batch_x) == 0:
                break  # avoid empty batch error

            y_true_batch = np.argmax(batch_y, axis=1)
            y_true.extend(y_true_batch)

            preds = model.predict(batch_x, verbose=0)  # suppress progress bar
            y_pred_batch = np.argmax(preds, axis=1)
            y_pred.extend(y_pred_batch)

            if preds.shape[1] == 2:
                y_prob_batch = preds[:, 1]
            else:
                y_prob_batch = np.max(preds, axis=1)
            y_prob.extend(y_prob_batch)
    except StopIteration:
        print("Test generator exhausted.")
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return

    if not y_true:
        print("No data to evaluate.")
        return

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    cm = confusion_matrix(y_true, y_pred)

    label_info = f" ({labelType.name})" if labelType else ""

    print(f"\n=== Classification Report{label_info} ===")
    print(classification_report(y_true, y_pred))
    print("Confusion Matrix:")
    print(cm)
    print(f"Overall Accuracy:      {acc:.4f}")
    print(f"Weighted Precision:    {prec:.4f}")
    print(f"Weighted Recall:       {rec:.4f}")
    print(f"Weighted F1-score:     {f1:.4f}")
    print(f"len(y_prob):           {len(y_prob)}")

    if len(set(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        print(f"ROC AUC:               {roc_auc:.4f}")

        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.title(f"ROC Curve{label_info}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.show()
