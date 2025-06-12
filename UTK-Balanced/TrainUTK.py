import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from util import UTKLabelType , removeCorruptImages
from resnet18model import ResNet18Model
from dataset import getGenerators , extractCSVLabels ,ageToBin

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    auc,
    classification_report
)
import matplotlib.pyplot as plt 

if __name__ == "__main__":
    # removeCorruptImages()
    label =  UTKLabelType.AGE #can change for age or whatever
    model = ResNet18Model(label)

    if label == UTKLabelType.AGE:
        title = 'Age'
    elif label == UTKLabelType.GENDER:
        title = 'Gender'
    else:
        title = 'Race'

    # #get training and testing data
    csvPath = f"./UTK/balancedAge2.csv"
    print(csvPath)
    filepaths, label_data = extractCSVLabels(csvPath,label)

    if label == UTKLabelType.AGE:
        label_data = [ageToBin(int(age)) for age in label_data]

    print(f"Unique classes in label_data: {sorted(set(label_data))}")

    train_loader , test_loader = getGenerators( filepaths, label_data ,label)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    num_epochs = 10  # Number of epochs for training

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move inputs and labels to GPU if available
            
            optimizer.zero_grad()  # Zero the parameter gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Calculate loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update parameters
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")
    y_true, y_pred, y_prob = [], [], []
    model.eval()  # Set the model to evaluation mode
    val_correct = 0
    val_total = 0
    with torch.no_grad():  # No need to compute gradients for validation
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)
            # accumulate
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            # for binary: probability of “class=1” 
            # (if you have more than 2 classes you can skip ROC or do one-vs-rest)
            if probs.size(1) == 2:
                y_prob.extend(probs[:,1].cpu().tolist())

    acc   = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted"
    )
    cm    = confusion_matrix(y_true, y_pred)

    print(f"\n=== {title} Classification Report ===")
    print(classification_report(y_true, y_pred))
    print("Confusion Matrix:")
    print(cm)
    print(f"Overall Accuracy: {acc:.4f}")
    print(f"Weighted Precision: {prec:.4f}")
    print(f"Weighted Recall:    {rec:.4f}")
    print(f"Weighted F1-score:  {f1:.4f}")
    print(f"len(y_prob): {len(y_prob)}")
    # 3) optional: ROC & AUC for binary
    if len(set(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc     = auc(fpr, tpr)
        print(f"ROC AUC:           {roc_auc:.4f}")

        # if you want to plot it:
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0,1], [0,1], linestyle="--", color="gray")
        plt.title(f"{title} ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.show()

    # # Print model summary (optional)
    # print(model)
    torch.save(model.state_dict(), ('resnet18_'+title+'Balanced.pth'))
