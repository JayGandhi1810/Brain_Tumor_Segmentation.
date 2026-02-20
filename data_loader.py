
import os
import numpy as np
import cv2
from glob import glob
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random

import config


# Helper: create output directories
def create_directories():
    for d in [config.MODEL_SAVE_PATH, config.RESULTS_PATH,
              config.PREDICTIONS_PATH, config.VISUALIZATIONS_PATH,
              config.LOGS_PATH]:
        os.makedirs(d, exist_ok=True)
        print(f"  Directory ready: {d}")


# Collect file paths from numbered subfolders
def collect_file_paths(images_base, masks_base):
    image_files, mask_files = [], []

    # Only use folders specified in config
    target_folders = config.TRAINING_FOLDERS

    print(f"\nLoading tumor classes:")
    for folder in target_folders:
        class_name = config.TUMOR_CLASSES.get(folder, f'Class {folder}')
        img_dir  = os.path.join(images_base, folder)
        mask_dir = os.path.join(masks_base,  folder)

        if not os.path.exists(img_dir):
            print(f"  ⚠️  Folder {folder} not found, skipping")
            continue

        imgs  = sorted(glob(os.path.join(img_dir,  "*.jpg")))
        imgs += sorted(glob(os.path.join(img_dir,  "*.png")))

        msks  = sorted(glob(os.path.join(mask_dir, "*_m.jpg")))
        msks += sorted(glob(os.path.join(mask_dir, "*_m.png")))
        if not msks:
            msks  = sorted(glob(os.path.join(mask_dir, "*.jpg")))
            msks += sorted(glob(os.path.join(mask_dir, "*.png")))

        # Match counts
        n = min(len(imgs), len(msks))
        imgs = imgs[:n]
        msks = msks[:n]

        image_files.extend(imgs)
        mask_files.extend(msks)

        print(f"  ✅ Folder {folder} ({class_name}): {n} pairs loaded")

    print(f"\n  Folders skipped : "
          f"{[f'{k}({v})' for k,v in config.TUMOR_CLASSES.items() if k not in target_folders]}")
    print(f"  Total pairs     : {len(image_files)}")

    return sorted(image_files), sorted(mask_files)


# PyTorch Dataset

class BrainTumorDataset(Dataset):
    """Custom Dataset for Brain Tumor Segmentation"""

    def __init__(self, image_paths, mask_paths, augment=False):
        self.image_paths = image_paths
        self.mask_paths  = mask_paths
        self.augment     = augment

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (config.IMG_WIDTH, config.IMG_HEIGHT))
        img = img.astype(np.float32) / 255.0
        
        msk = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        msk = cv2.resize(msk, (config.IMG_WIDTH, config.IMG_HEIGHT))
        msk = (msk.astype(np.float32) / 255.0 > 0.5).astype(np.float32)

        img = torch.from_numpy(img).permute(2, 0, 1)   # (C, H, W)
        msk = torch.from_numpy(msk).unsqueeze(0)        # (1, H, W)

        if self.augment and config.AUGMENTATION:
            img, msk = self._augment(img, msk)

        return img, msk

    def _augment(self, img, msk):
        """Apply identical random transforms to image and mask"""
        # Random horizontal flip
        if config.HORIZONTAL_FLIP and random.random() > 0.5:
            img = TF.hflip(img)
            msk = TF.hflip(msk)

        # Random vertical flip
        if config.VERTICAL_FLIP and random.random() > 0.5:
            img = TF.vflip(img)
            msk = TF.vflip(msk)

        # Random rotation
        if random.random() > 0.5:
            angle = random.uniform(-config.ROTATION_DEGREES,
                                    config.ROTATION_DEGREES)
            img = TF.rotate(img, angle)
            msk = TF.rotate(msk, angle)

        # Random brightness / contrast (image only)
        if random.random() > 0.5:
            img = TF.adjust_brightness(img, random.uniform(0.8, 1.2))
            img = TF.adjust_contrast(img,  random.uniform(0.8, 1.2))

        return img, msk


# Build DataLoaders

def prepare_data():
    print("=" * 55)
    print("DATA PREPARATION")
    print("=" * 55)

    torch.manual_seed(config.RANDOM_SEED)
    create_directories()

    # Collect paths
    image_files, mask_files = collect_file_paths(
        config.IMAGES_BASE_PATH, config.MASKS_BASE_PATH)

    if len(image_files) == 0:
        raise ValueError("No images found! Check DATASET_PATH in config.py")

    # Split
    X_tr, X_val, y_tr, y_val = train_test_split(
        image_files, mask_files,
        test_size=config.VALIDATION_SPLIT,
        random_state=config.RANDOM_SEED)

    print(f"\nTraining samples  : {len(X_tr)}")
    print(f"Validation samples: {len(X_val)}")

    # Datasets
    train_ds = BrainTumorDataset(X_tr,  y_tr,  augment=True)
    val_ds   = BrainTumorDataset(X_val, y_val, augment=False)

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE,
                              shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=config.BATCH_SIZE,
                              shuffle=False, num_workers=0, pin_memory=True)

    print(f"\nBatch size: {config.BATCH_SIZE}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val   batches: {len(val_loader)}")
    print("=" * 55)

    return train_loader, val_loader, X_val, y_val
