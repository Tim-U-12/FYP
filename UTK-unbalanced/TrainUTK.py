import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from util import UTKLabelType , removeCorruptImages
from resnet18model import ResNet18Model
from dataset import getGenerators

if __name__ == "__main__":
    removeCorruptImages
    label =  UTKLabelType.GENDER #can change for age or whatever
    model = ResNet18Model(label)

    if label == UTKLabelType.AGE:
        title = 'Age'
    elif label == UTKLabelType.GENDER:
        title = 'Gender'
    else:
        title = 'Race'

    #get training and testing data
    train_loader , test_loader = getGenerators(label)

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

    model.eval()  # Set the model to evaluation mode
    val_correct = 0
    val_total = 0
    with torch.no_grad():  # No need to compute gradients for validation
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_accuracy = 100 * val_correct / val_total
    print(f"Validation Accuracy: {val_accuracy:.2f}%")

    # Print model summary (optional)
    print(model)
    torch.save(model.state_dict(), ('resnet18_'+title+'.pth'))
