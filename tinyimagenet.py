from datasets import load_dataset
from PIL import Image
import os
from tqdm import tqdm

# 1️⃣ Load dataset
print("🔽 Loading Tiny ImageNet from Hugging Face...")
dataset = load_dataset("zh-plus/tiny-imagenet")

# 2️⃣ Define output directories
output_root = "/content/tiny-imagenet-200"
os.makedirs(output_root, exist_ok=True)

# 3️⃣ Function to save split
def save_split(split_name):
    print(f"\n📦 Processing {split_name} split...")
    split_dir = os.path.join(output_root, split_name)
    os.makedirs(split_dir, exist_ok=True)

    split_data = dataset[split_name]

    # Loop through dataset
    for idx, example in enumerate(tqdm(split_data, total=len(split_data))):
        img = example["image"]
        label = example["label"]
        class_name = split_data.features["label"].int2str(label)

        # Create class directory
        class_dir = os.path.join(split_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        # Convert to RGB (fix grayscale)
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Generate unique filename
        img_path = os.path.join(class_dir, f"{split_name}_{idx}.jpg")

        # Save image
        img.save(img_path, "JPEG")

    print(f"✅ Saved {split_name} split to {split_dir}")

# 4️⃣ Process both splits
save_split("train")
save_split("valid")

# 5️⃣ Quick summary
train_count = sum(len(files) for _, _, files in os.walk(os.path.join(output_root, "train")))
val_count = sum(len(files) for _, _, files in os.walk(os.path.join(output_root, "valid")))
print(f"\n✅ Done! Train images: {train_count}, Val images: {val_count}")
