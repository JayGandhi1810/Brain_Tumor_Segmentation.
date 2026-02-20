"""
Training Module
"""

import os
import time
import csv
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

import config
from model import dice_coefficient, iou_score

# One epoch

def run_epoch(model, loader, criterion, optimizer, device, training=True):
    model.train() if training else model.eval()

    total_loss, total_dice, total_iou = 0., 0., 0.
    n = len(loader)

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        bar = tqdm(loader, desc="Train" if training else "Val  ", leave=False)
        for imgs, masks in bar:
            imgs  = imgs.to(device)
            masks = masks.to(device)

            preds = model(imgs)
            loss  = criterion(preds, masks)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_dice += dice_coefficient(preds, masks).item()
            total_iou  += iou_score(preds, masks).item()

            bar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss/n, total_dice/n, total_iou/n


# Training loop

def train_model(model, criterion, optimizer, scheduler,
                train_loader, val_loader, device,
                model_name='unet_brain_tumor'):

    print("=" * 55)
    print("MODEL TRAINING")
    print("=" * 55)
    print(f"Device        : {device}")
    print(f"Epochs        : {config.EPOCHS}")
    print(f"Batch size    : {config.BATCH_SIZE}")
    print(f"Learning rate : {config.LEARNING_RATE}")
    print(f"Train batches : {len(train_loader)}")
    print(f"Val   batches : {len(val_loader)}")
    print("=" * 55)

    best_val_dice = 0.0
    patience_cnt  = 0
    best_path     = os.path.join(config.MODEL_SAVE_PATH, f'{model_name}_best.pth')

    # CSV log
    log_path = os.path.join(config.LOGS_PATH, f'{model_name}_log.csv')
    csv_file = open(log_path, 'w', newline='')
    writer   = csv.writer(csv_file)
    writer.writerow(['epoch','train_loss','train_dice','train_iou',
                              'val_loss',  'val_dice',  'val_iou', 'lr'])

    history = {'train_loss':[], 'train_dice':[], 'train_iou':[],
               'val_loss':[],   'val_dice':[],   'val_iou':[]}

    start_time = time.time()

    for epoch in range(1, config.EPOCHS + 1):
        ep_start = time.time()

        # Train
        tr_loss, tr_dice, tr_iou = run_epoch(
            model, train_loader, criterion, optimizer, device, training=True)

        # Validate
        vl_loss, vl_dice, vl_iou = run_epoch(
            model, val_loader, criterion, optimizer, device, training=False)

        ep_time = time.time() - ep_start
        lr_now  = optimizer.param_groups[0]['lr']

        # Log
        for k, v in zip(history.keys(),
                        [tr_loss, tr_dice, tr_iou, vl_loss, vl_dice, vl_iou]):
            history[k].append(v)

        writer.writerow([epoch, tr_loss, tr_dice, tr_iou,
                                vl_loss, vl_dice, vl_iou, lr_now])
        csv_file.flush()

        # Print
        print(f"Epoch [{epoch:3d}/{config.EPOCHS}] "
              f"| {ep_time:5.1f}s "
              f"| Loss: {tr_loss:.4f}/{vl_loss:.4f} "
              f"| Dice: {tr_dice:.4f}/{vl_dice:.4f} "
              f"| IoU: {tr_iou:.4f}/{vl_iou:.4f} "
              f"| LR: {lr_now:.2e}")

        # Scheduler
        scheduler.step(vl_dice)

        # Checkpoint
        if vl_dice > best_val_dice:
            best_val_dice = vl_dice
            patience_cnt  = 0
            torch.save(model.state_dict(), best_path)
            print(f"  ✅ Best model saved! Val Dice = {best_val_dice:.4f}")
        else:
            patience_cnt += 1
            print(f"  No improvement ({patience_cnt}/{config.EARLY_STOPPING_PATIENCE})")

        # Early stopping
        if patience_cnt >= config.EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch}!")
            break

    csv_file.close()
    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time/60:.1f} minutes")
    print(f"Best Val Dice: {best_val_dice:.4f}")
    print(f"Model saved : {best_path}")

    # Save final model
    final_path = os.path.join(config.MODEL_SAVE_PATH, f'{model_name}_final.pth')
    torch.save(model.state_dict(), final_path)

    # Plot history
    plot_path = os.path.join(config.VISUALIZATIONS_PATH,
                             f'{model_name}_training_history.png')
    plot_history(history, save_path=plot_path)

    return history, best_path


# Plot training curves

def plot_history(history, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Training History', fontsize=14)

    pairs = [('Loss',  'train_loss', 'val_loss'),
             ('Dice',  'train_dice', 'val_dice'),
             ('IoU',   'train_iou',  'val_iou')]

    for ax, (title, tr_key, vl_key) in zip(axes, pairs):
        epochs = range(1, len(history[tr_key]) + 1)
        ax.plot(epochs, history[tr_key], label='Train')
        ax.plot(epochs, history[vl_key], label='Val')
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history saved: {save_path}")
    plt.show()
