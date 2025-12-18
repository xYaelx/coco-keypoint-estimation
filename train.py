import torch
from visualization import save_sample

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    
    for batch_idx, (images, heatmaps) in enumerate(dataloader):
        images = images.to(device)
        heatmaps = heatmaps.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        
        # Compute loss
        loss = criterion(outputs, heatmaps)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f'Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}')
    
    return running_loss / len(dataloader)


def validate(model, dataloader, criterion, device, epoch=0, save_vis=False):
    """Validate the model and optionally save visualizations"""
    model.eval()
    running_loss = 0.0
    batch_count = 0
    
    with torch.no_grad():
        for batch_idx, (images, heatmaps) in enumerate(dataloader):
            images = images.to(device)
            heatmaps = heatmaps.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, heatmaps)
            
            running_loss += loss.item()
            batch_count += 1
            
            # Save 1 image per epoch for visualizing
            if save_vis and batch_idx == 1:
                filename = f'out_vis/epoch_{epoch:03d}.jpg'
                save_sample(images[0].cpu(), outputs[0].cpu(), heatmaps[0].cpu(), filename)
    
    return running_loss / batch_count
