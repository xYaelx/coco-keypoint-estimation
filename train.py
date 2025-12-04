import torch

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


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for images, heatmaps in dataloader:
            images = images.to(device)
            heatmaps = heatmaps.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, heatmaps)
            
            running_loss += loss.item()
    
    return running_loss / len(dataloader)
