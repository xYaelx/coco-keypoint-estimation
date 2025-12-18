import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from config import (
    ROOT_DIR, USE_RAW, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE,
    IMG_SIZE, HEATMAP_SIZE, NUM_KEYPOINTS, NUM_WORKERS,
    LR_SCHEDULER_STEP_SIZE, LR_SCHEDULER_GAMMA, CHECKPOINT_DIR, MODEL_SAVE_NAME,
    MAX_SAMPLES, VIS_INTERVAL
)
from dataset import COCOKeypointDataset
from model import KeypointEstimationModel
from loss import KeypointMSELoss
from train import train_epoch, validate
from utils import create_checkpoint_dir


def collate_fn(batch):
    """Filter out None batches to skip images without keypoints"""
    batch = [item for item in batch if item[0] is not None]
    if len(batch) == 0:
        return None
    return torch.stack([item[0] for item in batch]), torch.stack([item[1] for item in batch])


def main():
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create checkpoint directory
    create_checkpoint_dir(CHECKPOINT_DIR)
    
    # Create datasets
    print('Loading datasets...')
    train_dataset = COCOKeypointDataset(
        ROOT_DIR, 
        split='train',
        img_size=IMG_SIZE,
        heatmap_size=HEATMAP_SIZE,
        use_raw=USE_RAW,
        max_samples=MAX_SAMPLES
    )
    
    val_dataset = COCOKeypointDataset(
        ROOT_DIR,
        split='validation', 
        img_size=IMG_SIZE,
        heatmap_size=HEATMAP_SIZE,
        use_raw=USE_RAW,
        max_samples=MAX_SAMPLES
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )
    
    # Initialize model
    print('Initializing model...')
    model = KeypointEstimationModel(num_keypoints=NUM_KEYPOINTS, pretrained=True)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = KeypointMSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_SCHEDULER_STEP_SIZE, gamma=LR_SCHEDULER_GAMMA)
    
    # Training loop
    print('Starting training...')
    best_val_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        print(f'\nEpoch {epoch+1}/{NUM_EPOCHS}')
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate and visualize every VIS_INTERVAL epochs
        should_visualize = (epoch + 1) % VIS_INTERVAL == 0
        val_loss = validate(model, val_loader, criterion, device, epoch=epoch, save_vis=should_visualize)
        
        # Update learning rate
        scheduler.step()
        
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = f'{CHECKPOINT_DIR}/{MODEL_SAVE_NAME}'
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Saved best model to {checkpoint_path}!')
    
    print('Training complete!')


if __name__ == '__main__':
    main()