import torch
from torchvision import models, transforms, datasets
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
import torch
from torchvision import models
# Import ResNet18_Weights
from torchvision.models import ResNet18_Weights

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_folder = 'archive/Gender/test'

test_dataset = datasets.ImageFolder(root = test_folder , transform = transform)

test_loader = DataLoader( test_dataset , batch_size = 32 , shuffle = False)

# Define the model architecture again
model = models.resnet18(weights=None) 


for param in model.parameters():
    param.requires_grad = True

for param in model.fc.parameters():
   param.requires_grad = True

# Load the saved model weights
num_classes = len(test_dataset.classes)
#model.fc =nn.Sequential( nn.Dropout(0.5), nn.Linear(model.fc.in_features, num_classes))
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load('resnet18_Gender.pth'))

# Set the model to evaluation mode
model.eval()

#gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

correct = 0
total = 0 

with torch.no_grad():  # No need to compute gradients for inference
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Move to device (GPU or CPU)
        
        # Forward pass
        outputs = model(inputs)
        
        # Get the predicted class (the one with the highest probability)
        _, predicted = torch.max(outputs, 1)
        
        # Calculate accuracy
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Test Accuracy: {accuracy:.2f}%')
