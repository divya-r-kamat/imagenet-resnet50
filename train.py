import torch
import random
from utils_regularization import cutmix_data
from utils import EarlyStopping

from tqdm import tqdm
import torch
import random
from utils_regularization import cutmix_data

def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, device, epoch, use_cutmix=True):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch} [Train]", ncols=100)

    for batch_idx, (inputs, targets) in loop:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        apply_cutmix = use_cutmix and random.random() < 0.5
        if apply_cutmix:
            inputs, targets_a, targets_b, lam = cutmix_data(inputs.cpu(), targets.cpu())
            inputs = inputs.to(device)
            targets_a = targets_a.to(device)
            targets_b = targets_b.to(device)
            outputs = model(inputs)
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)

        if apply_cutmix:
            correct += lam * predicted.eq(targets_a).sum().item() + (1 - lam) * predicted.eq(targets_b).sum().item()
        else:
            correct += predicted.eq(targets).sum().item()

        # Update tqdm description dynamically
        loop.set_postfix(
            loss=f"{running_loss / total:.4f}",
            acc=f"{100. * correct / total:.2f}%"
        )

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc



def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss = running_loss / total
    val_acc = 100. * correct / total
    print(f"Validation: Average loss: {val_loss:.4f}, Accuracy: {correct}/{total} ({val_acc:.2f}%)")
    return val_loss, val_acc
