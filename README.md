# COCO Keypoint Estimation

A PyTorch framework for human keypoint estimation on the COCO dataset, designed for research and development of pose estimation models.

## Overview

This project implements a modular keypoint estimation system that:
- Uses ResNet50 with deconvolutional layers to predict heatmaps for 17 human keypoints
- Loads and preprocesses COCO 2017 dataset with automatic heatmap generation
- Trains models with MSE loss and learning rate scheduling
- Supports checkpoint saving and validation monitoring

## Features

- **Model Architecture**: ResNet50 backbone with 3 deconvolutional layers for upsampling
- **Dataset Pipeline**: COCO keypoint dataset loading with Gaussian heatmap generation
- **Training Loop**: Flexible training/validation with best model checkpointing
- **Configuration**: Centralized config for easy hyperparameter tuning
- **Experiment Tracking**: Integration with Weights & Biases (WandB)

## Installation

### Prerequisites
- Python 3.11+
- CUDA 12.1 (GPU support)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd coco-keypoint-estimation
```

2. Create and activate a virtual environment:
```bash
python -m venv cocoPy
cocoPy\Scripts\activate  # On Windows
# or
source cocoPy/bin/activate  # On Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

Or using the project configuration:
```bash
pip install -e .
```

### Dependencies
- torch >= 2.9.0
- torchvision >= 0.24.0
- pycocotools >= 2.0.10
- fiftyone >= 1.9.0
- wandb >= 0.22.3

## Dataset Setup

1. **Download COCO 2017 dataset** using FiftyOne:
```python
import fiftyone as fo
dataset = fo.zoo.load_zoo_dataset("coco-2017", split=["train", "validation"])
```

2. **Update the data path** in `config.py`:
```python
ROOT_DIR = r'C:\Users\beyae\fiftyone\coco-2017'  # Your COCO dataset path
```

The dataset should have the following structure:
```
coco-2017/
├── train/
│   ├── data/          # Training images
│   └── labels.json    # FiftyOne labels (optional)
├── validation/
│   ├── data/          # Validation images
│   └── labels.json    # FiftyOne labels (optional)
└── raw/
    ├── person_keypoints_train2017.json
    └── person_keypoints_val2017.json
```

## Configuration

Edit `config.py` to customize:

```python
# Data paths
ROOT_DIR = r'C:\Users\beyae\fiftyone\coco-2017'
USE_RAW = True  # Use original COCO annotations

# Training hyperparameters
BATCH_SIZE = 8
NUM_EPOCHS = 25
LEARNING_RATE = 1e-3

# Model hyperparameters
IMG_SIZE = 256          # Input image resolution
HEATMAP_SIZE = 64       # Output heatmap resolution
NUM_KEYPOINTS = 17      # COCO has 17 keypoints per person

# Training settings
NUM_WORKERS = 4
LR_SCHEDULER_STEP_SIZE = 10
LR_SCHEDULER_GAMMA = 0.5

# Checkpointing
CHECKPOINT_DIR = 'checkpoints'
MODEL_SAVE_NAME = 'best_keypoint_model.pth'
```

## Usage

### Training

Run the training script:

```bash
python main.py
```

The training will:
1. Load and preprocess the COCO dataset
2. Initialize a ResNet50-based keypoint estimation model
3. Train for the specified number of epochs
4. Save the best model based on validation loss
5. Display training and validation metrics

**Example output:**
```
Using device: cuda
Loading datasets...
Found 64115 images with keypoint annotations
Initializing model...
Starting training...

Epoch 1/25
Train Loss: 0.0324, Val Loss: 0.0218
Saved best model to checkpoints/best_keypoint_model.pth!
```

### Model Architecture

The model consists of:
- **Backbone**: ResNet50 pretrained on ImageNet (removes final classification layers)
- **Upsampling**: 3 transposed convolution layers (8x upsampling total)
- **Output**: 17 heatmaps at 1/4 resolution of input

```
Input (B, 3, 256, 256)
    ↓
ResNet50 Backbone: (B, 2048, 8, 8)
    ↓
Upsample 1: (B, 1024, 16, 16)
    ↓
Upsample 2: (B, 512, 32, 32)
    ↓
Upsample 3: (B, 256, 64, 64)
    ↓
Output Layer: (B, 17, 64, 64)
```

### Dataset

The COCOKeypointDataset class:
- Loads images and COCO keypoint annotations
- Resizes images to 256×256
- Generates Gaussian heatmaps (σ=2) for visible keypoints
- Handles missing or crowded annotations gracefully
- Supports dataset size limiting with `MAX_SAMPLES`

**COCO Keypoints (17 total):**
1. Nose, 2. Left Eye, 3. Right Eye, 4. Left Ear, 5. Right Ear
6. Left Shoulder, 7. Right Shoulder, 8. Left Elbow, 9. Right Elbow
10. Left Wrist, 11. Right Wrist, 12. Left Hip, 13. Right Hip
14. Left Knee, 15. Right Knee, 16. Left Ankle, 17. Right Ankle

### Loss Function

The KeypointMSELoss computes Mean Squared Error between predicted and ground-truth heatmaps, supporting per-keypoint weighting for imbalanced visibility.

## Project Structure

```
coco-keypoint-estimation/
├── main.py              # Entry point for training
├── config.py            # Hyperparameters and paths
├── model.py             # ResNet50-based model
├── dataset.py           # COCO dataset loader
├── loss.py              # Loss functions
├── train.py             # Training and validation loops
├── utils.py             # Helper utilities
├── checkpoints/         # Saved model checkpoints
├── out_vis/             # Output visualizations
├── wandb/               # WandB experiment logs
└── README.md            # This file
```

## Results

After training for 25 epochs with the default config:
- Best validation loss achieved: ~0.02
- Training time: ~2-4 hours on GPU (RTX 3090 or similar)
- Model checkpoint size: ~100 MB

## Future Work

- [ ] Add COCO evaluation metrics (mAP, OKS)
- [ ] Implement additional model architectures (HRNet, MobileNet)
- [ ] Add data augmentation (rotation, color jitter, random flip)
- [ ] Visualization tools for predicted vs. ground-truth keypoints
- [ ] ONNX model export for deployment
- [ ] Mixed precision training
- [ ] Distributed training support

## Notes

- The model uses pretrained ResNet50 weights from ImageNet
- Heatmap targets use Gaussian distributions with σ=2
- Learning rate is reduced by 0.5× every 10 epochs
- Best model is automatically saved during training

## License

See `LICENSE` for details.