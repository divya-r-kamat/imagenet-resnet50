# main.py
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import OneCycleLR
from dataset import get_dataloaders
from model import get_resnet50
from train import train_one_epoch, validate
from utils_regularization import LabelSmoothingCrossEntropy
from utils_plot import plot_loss_accuracy
from utils import EarlyStopping
from config import CONFIG

def main():
    device = CONFIG["device"]

    # ---------------------------
    # Data loaders
    # ---------------------------
    train_loader, val_loader = get_dataloaders(CONFIG["data_dir"], CONFIG["image_size"], CONFIG["batch_size"], CONFIG["num_workers"])

    # ---------------------------
    # Model
    # ---------------------------
    model = get_resnet50(num_classes=CONFIG["num_classes"]).to(device)

    # ---------------------------
    # Loss, Optimizer, Scheduler
    # ---------------------------
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    optimizer = SGD(model.parameters(), lr=CONFIG["lr"], momentum=CONFIG["momentum"], weight_decay=CONFIG["weight_decay"])
    scheduler = OneCycleLR(optimizer, max_lr=CONFIG["lr"], epochs=CONFIG["epochs"], steps_per_epoch=len(train_loader),
                           pct_start=0.3, div_factor=10, final_div_factor=100)

    # ---------------------------
    # Early stopping
    # ---------------------------
    early_stopper = EarlyStopping(patience=CONFIG["patience"], path="best_model.pth")

    # ---------------------------
    # Logs
    # ---------------------------
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(1, CONFIG["epochs"]+1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        early_stopper(val_acc, model)
        if early_stopper.early_stop:
            print("Early stopping triggered. Training stopped.")
            break

    # ---------------------------
    # Save plots
    # ---------------------------
    plot_loss_accuracy(train_losses, val_losses, train_accs, val_accs, save_path="train_val_plot.png")
    print("Training completed. Best model saved as 'best_model.pth'.")

if __name__ == "__main__":
    main()
