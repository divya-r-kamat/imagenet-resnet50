# dataset.py
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import datasets
from torch.utils.data import DataLoader
import numpy as np


def get_train_transforms(image_size):
    return A.Compose([
        A.RandomResizedCrop(size=(image_size, image_size), scale=(0.7, 1.0), ratio=(0.75, 1.33)),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.ColorJitter(0.4, 0.4, 0.4, 0.2, p=0.5),
        A.CoarseDropout(max_holes=1, max_height=int(0.2*image_size), max_width=int(0.2*image_size), p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def get_valid_transforms(image_size):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


class AlbumentationsDataset(datasets.ImageFolder):
    """ImageFolder wrapper to apply Albumentations transforms."""
    def __init__(self, root, transform=None):
        super().__init__(root)
        self.albumentations_transform = transform

    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)
        image = np.array(image)
        if self.albumentations_transform:
            augmented = self.albumentations_transform(image=image)
            image = augmented["image"]
        return image, label


def get_dataloaders(data_dir, image_size, batch_size, num_workers):
    import numpy as np

    train_tfms = get_train_transforms(image_size)
    val_tfms = get_valid_transforms(image_size)

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val") if os.path.exists(os.path.join(data_dir, "val")) else os.path.join(data_dir, "valid")

    train_ds = AlbumentationsDataset(train_dir, transform=train_tfms)
    val_ds = AlbumentationsDataset(val_dir, transform=val_tfms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader
