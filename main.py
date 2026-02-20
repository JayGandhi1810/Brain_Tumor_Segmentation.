"""
Main Execution Script - Brain Tumor Segmentation
"""

import sys
import torch
import config


# GPU Configuration

def configure_device():
    print("=" * 55)
    print("DEVICE CONFIGURATION")
    print("=" * 55)

    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu    = torch.cuda.get_device_properties(0)
        print(f"  GPU detected!")
        print(f"   Name    : {gpu.name}")
        print(f"   Memory  : {gpu.total_memory / 1024**3:.1f} GB")
        print(f"   CUDA    : {torch.version.cuda}")

        # Allow gradual memory growth
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')
        print("⚠️  No GPU detected — using CPU (training will be slow)")

    print(f"   Device  : {device}")
    print("=" * 55)
    return device


def print_system_info():
    print("\n" + "=" * 55)
    print("SYSTEM INFORMATION")
    print("=" * 55)
    print(f"Python  : {sys.version[:6]}")
    print(f"PyTorch : {torch.__version__}")
    print(f"CUDA    : {torch.version.cuda}")
    print(f"cuDNN   : {torch.backends.cudnn.version()}")
    print("=" * 55 + "\n")


# Main pipeline

def main():
    print("\n")
    print("*" * 55)
    print("*" + " " * 53 + "*")
    print("*   BRAIN TUMOR SEGMENTATION (PyTorch + GPU)      *")
    print("*" + " " * 53 + "*")
    print("*" * 55 + "\n")

    # Step 0: Setup
    print_system_info()
    device = configure_device()

    torch.manual_seed(config.RANDOM_SEED)
    if device.type == 'cuda':
        torch.cuda.manual_seed(config.RANDOM_SEED)

    # Step 1: Data
    print("\nSTEP 1: DATA PREPARATION")
    print("-" * 40)
    from data_loader import prepare_data
    train_loader, val_loader, val_img_paths, val_mask_paths = prepare_data()

    # Step 2: Model
    print("\nSTEP 2: MODEL BUILDING")
    print("-" * 40)
    from model import get_model
    model, criterion, optimizer, scheduler = get_model(device)

    # Step 3: Training
    print("\nSTEP 3: TRAINING")
    print("-" * 40)
    from train import train_model
    history, best_model_path = train_model(
        model, criterion, optimizer, scheduler,
        train_loader, val_loader, device)

    # Step 4: Load best model
    print("\nLoading best model for evaluation...")
    from model import load_model
    model = load_model(best_model_path, device)

    # Step 5: Evaluation
    print("\nSTEP 4: EVALUATION")
    print("-" * 40)
    from evaluate import comprehensive_evaluation
    metrics, _ = comprehensive_evaluation(
        model, val_img_paths, val_mask_paths, device)

    # Summary
    print("\n" + "*" * 55)
    print("PIPELINE COMPLETE!")
    print("*" * 55)
    print(f"\nFinal Metrics:")
    for name, val in metrics.items():
        print(f"  {name:<12}: {val:.4f}")
    print(f"\nOutputs saved to:")
    print(f"  Models       : {config.MODEL_SAVE_PATH}")
    print(f"  Predictions  : {config.PREDICTIONS_PATH}")
    print(f"  Visualizations: {config.VISUALIZATIONS_PATH}")
    print(f"  Logs         : {config.LOGS_PATH}")

    return model, history, metrics


# Predict a single image

def predict_single_image(model_path, image_path, device=None):
    """Run inference on one image and display result"""
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from model import load_model

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model(model_path, device)

    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (config.IMG_WIDTH, config.IMG_HEIGHT))

    t = torch.from_numpy(img.astype(np.float32) / 255.0) \
             .permute(2, 0, 1).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        pred = model(t).squeeze().cpu().numpy()

    pred_bin = (pred > 0.5).astype(np.float32)

    overlay = img.copy().astype(np.float32) / 255.0
    overlay[pred_bin == 1] = [1.0, 0.0, 0.0]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img);            axes[0].set_title('Input Image')
    axes[1].imshow(pred_bin, cmap='gray'); axes[1].set_title('Prediction')
    axes[2].imshow(overlay);        axes[2].set_title('Overlay')
    for ax in axes: ax.axis('off')
    plt.tight_layout()
    plt.show()

    return pred


# Entry point

if __name__ == "__main__":
    model, history, metrics = main()
