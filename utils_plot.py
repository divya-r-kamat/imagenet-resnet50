# utils_plot.py
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch
import numpy as np

# Plot Train/Val curves
def plot_loss_accuracy(train_losses, val_losses, train_accs, val_accs, save_path="train_val_plot.png"):
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(train_accs, label="Train Acc")
    plt.plot(val_accs, label="Val Acc")
    plt.title("Accuracy")
    plt.legend()

    plt.savefig(save_path)
    plt.show()

# Grad-CAM
def generate_gradcam(model, img_tensor, target_layer, device="cuda"):
    model.eval()
    img_tensor = img_tensor.to(device)
    cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=(device=="cuda"))
    grayscale_cam = cam(input_tensor=img_tensor)[0]
    img_np = img_tensor[0].permute(1,2,0).cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
    plt.imshow(cam_image)
    plt.axis('off')
    plt.show()
