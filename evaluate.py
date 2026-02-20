"""
Evaluation Module
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

import config
from model import dice_coefficient, iou_score
from data_loader import BrainTumorDataset
from torch.utils.data import DataLoader


# Metrics

def compute_all_metrics(preds_np, masks_np, threshold=0.5):
    p = (preds_np > threshold).astype(np.float32).flatten()
    t = masks_np.astype(np.float32).flatten()

    inter = (p * t).sum()
    union = p.sum() + t.sum() - inter

    dice = (2 * inter + 1e-6) / (p.sum() + t.sum() + 1e-6)
    iou  = (inter + 1e-6) / (union + 1e-6)
    acc  = (p == t).mean()

    tp = inter
    fp = p.sum() - inter
    fn = t.sum() - inter

    prec   = (tp + 1e-6) / (tp + fp + 1e-6)
    recall = (tp + 1e-6) / (tp + fn + 1e-6)
    f1     = 2 * prec * recall / (prec + recall + 1e-6)

    return {'Dice': dice, 'IoU': iou, 'Accuracy': acc,
            'Precision': prec, 'Recall': recall, 'F1': f1}


# Predict on DataLoader

def predict_all(model, loader, device):
    model.eval()
    all_preds, all_masks = [], []

    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="Predicting"):
            imgs  = imgs.to(device)
            preds = model(imgs).cpu().numpy()
            all_preds.append(preds)
            all_masks.append(masks.numpy())

    preds = np.concatenate(all_preds, axis=0)   # (N, 1, H, W)
    masks = np.concatenate(all_masks, axis=0)

    return preds, masks


# Visualise predictions

def visualize_predictions(image_paths, mask_paths, model, device,
                           n=config.NUM_SAMPLES_TO_VISUALIZE,
                           save_path=None):
    n = min(n, len(image_paths))
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    model.eval()
    for i in range(n):
        # Load
        img  = cv2.cvtColor(cv2.imread(image_paths[i]), cv2.COLOR_BGR2RGB)
        img  = cv2.resize(img, (config.IMG_WIDTH, config.IMG_HEIGHT))
        msk  = cv2.imread(mask_paths[i], cv2.IMREAD_GRAYSCALE)
        msk  = cv2.resize(msk, (config.IMG_WIDTH, config.IMG_HEIGHT))
        msk  = (msk / 255.0 > 0.5).astype(np.float32)

        # Predict
        t = torch.from_numpy(img.astype(np.float32) / 255.0) \
                 .permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(t).squeeze().cpu().numpy()

        pred_bin = (pred > 0.5).astype(np.float32)

        # Overlay
        overlay = img.copy().astype(np.float32) / 255.0
        overlay[pred_bin == 1] = [1.0, 0.0, 0.0]

        titles = ['Original', 'Ground Truth', 'Prediction', 'Overlay']
        imgs_  = [img, msk, pred_bin, overlay]
        cmaps  = [None, 'gray', 'gray', None]

        for j, (im, title, cm) in enumerate(zip(imgs_, titles, cmaps)):
            axes[i, j].imshow(im, cmap=cm)
            axes[i, j].set_title(title if i == 0 else '')
            axes[i, j].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualisation saved: {save_path}")
    plt.show()


# Save binary masks to disk

def save_predictions(image_paths, preds_np, out_dir=config.PREDICTIONS_PATH):
    os.makedirs(out_dir, exist_ok=True)
    for i, (path, pred) in enumerate(zip(image_paths, preds_np)):
        binary = (pred.squeeze() > 0.5).astype(np.uint8) * 255

        # Load original for overlay
        img = cv2.imread(path)
        img = cv2.resize(img, (config.IMG_WIDTH, config.IMG_HEIGHT))

        mask_path   = os.path.join(out_dir, f'pred_mask_{i:04d}.png')
        overlay_path = os.path.join(out_dir, f'pred_overlay_{i:04d}.png')

        cv2.imwrite(mask_path, binary)

        overlay = img.copy()
        overlay[binary == 255] = [0, 0, 255]   # Red in BGR
        cv2.imwrite(overlay_path, overlay)

    print(f"Saved {len(preds_np)} predictions to {out_dir}")


# Plot metric bar chart

def plot_metrics(metrics, save_path=None):
    names  = list(metrics.keys())
    values = list(metrics.values())

    plt.figure(figsize=(10, 5))
    bars = plt.bar(names, values, color='steelblue', edgecolor='navy')
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                 f'{h:.4f}', ha='center', fontsize=9)
    plt.ylim(0, 1.12)
    plt.ylabel('Score')
    plt.title('Model Evaluation Metrics')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Metrics chart saved: {save_path}")
    plt.show()


# Full evaluation pipeline

def comprehensive_evaluation(model, val_image_paths, val_mask_paths, device):
    print("=" * 55)
    print("MODEL EVALUATION")
    print("=" * 55)

    # Build a loader for the validation set
    ds     = BrainTumorDataset(val_image_paths, val_mask_paths, augment=False)
    loader = DataLoader(ds, batch_size=config.BATCH_SIZE,
                        shuffle=False, num_workers=0)

    preds, masks = predict_all(model, loader, device)

    metrics = compute_all_metrics(preds, masks)

    print("\nEvaluation Results:")
    print("-" * 35)
    for name, val in metrics.items():
        print(f"  {name:<12}: {val:.4f}")
    print("-" * 35)

    # Visualise
    viz_path = os.path.join(config.VISUALIZATIONS_PATH,
                            'predictions_visualization.png')
    visualize_predictions(val_image_paths, val_mask_paths,
                          model, device, save_path=viz_path)

    # Save predictions to disk
    if config.SAVE_PREDICTIONS:
        save_predictions(val_image_paths, preds)

    # Metrics bar chart
    metrics_path = os.path.join(config.VISUALIZATIONS_PATH,
                                'metrics_comparison.png')
    plot_metrics(metrics, save_path=metrics_path)

    print("\n" + "=" * 55)
    print("EVALUATION COMPLETE")
    print("=" * 55)

    return metrics, preds
