# model.py
import torch.nn as nn
import torchvision.models as models

def get_resnet50(num_classes):
    model = models.resnet50(weights=None)  # train from scratch
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
