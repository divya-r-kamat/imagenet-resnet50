# model.py
import torch.nn as nn
import torchvision.models as models

def get_resnet50(num_classes=200, dropout=0.3):
    model = models.resnet50(weights=None)  # from scratch
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes)
    )
    return model
