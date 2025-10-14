# config.py
import torch
# config.py
CONFIG = {
    "data_dir": "/content/tiny-imagenet-200",   # change to /content/imagenet for full ImageNet-1k
    "num_classes": 200,                     # 1000 for ImageNet-1k
    "image_size": 64,                       # 224 for ImageNet-1k
    "batch_size": 128,
    "num_workers": 4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "lr": 0.1,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "epochs": 100,
    "patience": 10  # Early stopping patience
}

