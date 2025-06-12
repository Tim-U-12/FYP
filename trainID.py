import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import random_split, DataLoader
from torchvision import models
# Import ResNet18_Weights
from torchvision.models import ResNet18_Weights
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    auc,
    classification_report
)
import matplotlib.pyplot as plt 
import os

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

full_dataset = datasets.ImageFolder(root = 'archive/CelebrityFacesDataset/Photos', transform=transform)

train_pct = 0.8
n_total = len(full_dataset)
n_train = int(train_pct * n_total)
n_test  = n_total - n_train

# 3) random split
train_ds, test_ds = random_split(
    full_dataset,
    [n_train, n_test],
    generator=torch.Generator().manual_seed(42)
)


train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

num_classes = len(full_dataset.classes)
##
model.fc =nn.Sequential( nn.Linear(model.fc.in_features, num_classes))

for param in model.parameters():
    param.requires_grad = False

for param in model.fc.parameters():
   param.requires_grad = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

num_epochs = 20  # Number of epochs for training

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
    for inputs, labels in val_loader:
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
# val_accuracy = 100 * val_correct / val_total
# print(f"Validation Accuracy: {val_accuracy:.2f}%")

acc   = accuracy_score(y_true, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average="weighted"
)
cm    = confusion_matrix(y_true, y_pred)

print(f"\n=== ID Classification Report ===")
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
    plt.title(f"ID ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.show()



# Save the trained model
torch.save(model.state_dict(), 'resnet18_ID.pth')
