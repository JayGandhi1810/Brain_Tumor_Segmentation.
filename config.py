import os

# DIRECTORY PATHS

DATASET_PATH = r"C:\Users\gandh\Desktop\Folders\BSBI\Computer Vision and Airtificial Intelligence\20-02-2026\datasets\Brain Tumor Segmentation Dataset"

IMAGES_BASE_PATH = os.path.join(DATASET_PATH, "image")
MASKS_BASE_PATH  = os.path.join(DATASET_PATH, "mask")

# Output locations
MODEL_SAVE_PATH      = os.path.join(DATASET_PATH, "models")
RESULTS_PATH         = os.path.join(DATASET_PATH, "results")
PREDICTIONS_PATH     = os.path.join(DATASET_PATH, "predictions")
VISUALIZATIONS_PATH  = os.path.join(DATASET_PATH, "visualizations")
LOGS_PATH            = os.path.join(DATASET_PATH, "logs")
TUMOR_CLASSES = {
    '0': 'No Tumor',
    '1': 'Glioma',
    '2': 'Meningioma',
    '3': 'Pituitary'
}

# skip folder 0 - No Tumor
TRAINING_FOLDERS = ['1', '2', '3']

# IMAGE SETTINGS

IMG_HEIGHT   = 256
IMG_WIDTH    = 256
IMG_CHANNELS = 3

# TRAINING HYPERPARAMETERS

BATCH_SIZE       = 16      
EPOCHS           = 50
LEARNING_RATE    = 1e-4
VALIDATION_SPLIT = 0.2
RANDOM_SEED      = 42
NUM_WORKERS      = 4       

# Model
FILTERS      = [64, 128, 256, 512, 1024]
DROPOUT_RATE = 0.2

# Augmentation
AUGMENTATION       = True
ROTATION_DEGREES   = 20
HORIZONTAL_FLIP    = True
VERTICAL_FLIP      = True

# Callbacks
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE      = 5
REDUCE_LR_FACTOR        = 0.5
MIN_LR                  = 1e-7

# Checkpoint
SAVE_BEST_ONLY  = True
MONITOR_METRIC  = 'val_dice'

# Evaluation
NUM_SAMPLES_TO_VISUALIZE = 5
SAVE_PREDICTIONS         = True
