"""
Configuration for keypoint estimation training
"""
# Data paths
ROOT_DIR = r'C:\Users\beyae\fiftyone\coco-2017'  
USE_RAW = True  # Set to True to use raw/*.json files, False to use train/labels.json and validation/labels.json
MAX_SAMPLES = 2000  # dataset size for training/validation, None for full dataset

# Training hyperparameters
BATCH_SIZE = 10
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3

# Model hyperparameters
IMG_SIZE = 256
HEATMAP_SIZE = 64
NUM_KEYPOINTS = 17

# Training settings
NUM_WORKERS = 4
LR_SCHEDULER_STEP_SIZE = 10
LR_SCHEDULER_GAMMA = 0.5

# Checkpoint settings
CHECKPOINT_DIR = 'checkpoints'
MODEL_SAVE_NAME = 'best_keypoint_model.pth'
# Visualization settings
VIS_INTERVAL = 1  # Save 1 visualization image every X epochs