import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.optim import Adam
from util import UTKLabelType
from torchvision.models import ResNet18_Weights

def ResNet18Model(labelType: UTKLabelType):


    # Load pre-trained ResNet18 model without the fully connected layer
    base_model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        
    # Optionally freeze the base model layers for transfer learning
    for param in base_model.parameters():
        param.requires_grad = False
        
    # Modify the fully connected layer to match the output classes
    if labelType == UTKLabelType.AGE:
        base_model.fc = nn.Linear(base_model.fc.in_features, 8)  # 10 bins for age classification
    elif labelType == UTKLabelType.GENDER:
        base_model.fc = nn.Linear(base_model.fc.in_features, 2)  # 2 classes (Male/Female)
    elif labelType == UTKLabelType.RACE:
        base_model.fc = nn.Linear(base_model.fc.in_features, 5)  # 5 classes (race categories)
    else:
        raise ValueError("Invalid label type")


    return base_model


