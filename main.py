# main.py
import torch
import torch.nn as nn
import torch.optim as optim
from config import CONFIG
from dataset import get_dataloaders
from model import get_resnet50
from train import train_one_epoch, evaluate
from utils import EarlyStopping
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np

def main():
    device = CONFIG["device"]
    print(f"🔥 Training on: {device}")

    train_loader, val_loader = get_dataloaders(CONFIG["data_dir"], CONFIG["image_size"],
                                               CONFIG["batch_size"], CONFIG["num_workers"])

    model = get_resnet50(CONFIG["num_classes"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=CONFIG["lr"],
                          momentum=CONFIG["momentum"], weight_decay=CONFIG["weight_decay"])

    scheduler = OneCycleLR(optimizer, max_lr=CONFIG["lr"],
                           epochs=CONFIG["epochs"], steps_per_epoch=len(train_loader))

    early_stopper = EarlyStopping(patience=CONFIG["patience"], path="best_resnet50.pth")

    best_val_acc = 0
    for epoch in range(CONFIG["epochs"]):
        print(f"\n🌍 Epoch [{epoch+1}/{CONFIG['epochs']}]")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, scheduler)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc:.2f}%")

        early_stopper(val_acc, model)
        if early_stopper.early_stop:
            print("🏁 Early Stopping Triggered!")
            break

        best_val_acc = max(best_val_acc, val_acc)

    print(f"\n🎯 Best Validation Accuracy: {best_val_acc:.2f}%")

if __name__ == "__main__":
    main()
