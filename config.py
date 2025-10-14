# config.py
import torch

CONFIG = {
    # === dataset ===
    "data_dir": "/content/tiny-imagenet-200",   # change to "/data/imagenet" for full ImageNet
    "num_classes": 200,                  # 1000 for ImageNet
    "image_size": 64,                    # 224 for ImageNet-1K

    # === training ===
    "batch_size": 128,
    "epochs": 50,
    "lr": 0.1,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "patience": 10,                      # early stopping patience
    "num_workers": 8,

    # === device ===
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}
