import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet50
import numpy as np
import cv2
from pycocotools.coco import COCO
import os
import json

# ============================================================================
# 1. DATASET CLASS
# ============================================================================

class COCOKeypointDataset(Dataset):
    """COCO Keypoint Dataset for person keypoint detection"""
    
    def __init__(self, root_dir, split='train', img_size=256, heatmap_size=64, use_raw=True):
        """
        Args:
            root_dir: Root directory containing COCO data (e.g., C:\\Users\\beyae\\fiftyone\\coco-2017)
            split: 'train' or 'validation'
            img_size: Input image size
            heatmap_size: Output heatmap resolution
            use_raw: If True, use annotation files from raw folder, otherwise use labels.json
        """
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.heatmap_size = heatmap_size
        self.sigma = 2  # Gaussian sigma for heatmap generation
        
        # COCO has 17 keypoints for person
        self.num_keypoints = 17
        
        # raw is the original COOCO annotations while train/labels.json and validation/labels.json are custom by FiftyOne
        if split == 'train':
            self.img_dir = os.path.join(root_dir, 'train', 'data')
            if use_raw:
                self.ann_file = os.path.join(root_dir, 'raw', 'person_keypoints_train2017.json')
            else:
                self.ann_file = os.path.join(root_dir, 'train', 'labels.json')
        else:  # validation
            self.img_dir = os.path.join(root_dir, 'validation', 'data')
            if use_raw:
                self.ann_file = os.path.join(root_dir, 'raw', 'person_keypoints_val2017.json')
            else:
                self.ann_file = os.path.join(root_dir, 'validation', 'labels.json')
        
        # Load COCO annotations
        print(f'Loading annotations from {self.ann_file}...')
        self.coco = COCO(self.ann_file)
        
        # Get all person images with keypoints
        self.cat_ids = self.coco.getCatIds(catNms=['person'])
        self.img_ids = self.coco.getImgIds(catIds=self.cat_ids)
        
        # Filter images that have keypoint annotations
        self.valid_img_ids = []
        for img_id in self.img_ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.cat_ids, iscrowd=False)
            anns = self.coco.loadAnns(ann_ids)
            # Check if any annotation has keypoints
            if any('keypoints' in ann and ann['num_keypoints'] > 0 for ann in anns):
                self.valid_img_ids.append(img_id)
        
        print(f'Found {len(self.valid_img_ids)} images with keypoint annotations')
        
    def __len__(self):
        return len(self.valid_img_ids)
    
    def __getitem__(self, idx):
        img_id = self.valid_img_ids[idx]
        
        # Load image info
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not load image {img_path}")
            # Return dummy data
            img_tensor = torch.zeros(3, self.img_size, self.img_size)
            heatmaps = torch.zeros(self.num_keypoints, self.heatmap_size, self.heatmap_size)
            return img_tensor, heatmaps
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        # Get annotations for this image
        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.cat_ids, iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)
        
        # Find first annotation with keypoints
        keypoints = None
        for ann in anns:
            if 'keypoints' in ann and ann['num_keypoints'] > 0:
                keypoints = np.array(ann['keypoints']).reshape(-1, 3)
                break
        
        if keypoints is None:
            # Return dummy data if no keypoints found
            img_resized = cv2.resize(img, (self.img_size, self.img_size))
            img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
            heatmaps = torch.zeros(self.num_keypoints, self.heatmap_size, self.heatmap_size)
            return img_tensor, heatmaps
        
        # Resize image
        img_resized = cv2.resize(img, (self.img_size, self.img_size))
        
        # Scale keypoints - adjust coordinates according to resized image
        scale_x = self.img_size / w
        scale_y = self.img_size / h
        kpts = keypoints.copy().astype(np.float32)
        kpts[:, 0] *= scale_x
        kpts[:, 1] *= scale_y
        
        heatmaps = self.generate_heatmaps(kpts)
        
        # Convert to tensors
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        heatmaps_tensor = torch.from_numpy(heatmaps).float()
        
        return img_tensor, heatmaps_tensor
    
    def generate_heatmaps(self, keypoints):
        """Generate Gaussian heatmaps for keypoints"""
        heatmaps = np.zeros((self.num_keypoints, self.heatmap_size, self.heatmap_size))
        
        scale = self.heatmap_size / self.img_size
        
        for i, (x, y, v) in enumerate(keypoints):
            if v > 0:  # Only generate heatmap for visible keypoints
                x_hm = int(x * scale)
                y_hm = int(y * scale)
                
                if 0 <= x_hm < self.heatmap_size and 0 <= y_hm < self.heatmap_size:
                    # Generate 2D Gaussian
                    heatmap = self._generate_gaussian(self.heatmap_size, 
                                                      self.heatmap_size, 
                                                      x_hm, y_hm, self.sigma)
                    heatmaps[i] = np.maximum(heatmaps[i], heatmap)
        
        return heatmaps
    
    def _generate_gaussian(self, height, width, center_x, center_y, sigma):
        """Generate 2D Gaussian heatmap"""
        x = np.arange(0, width, 1, float)
        y = np.arange(0, height, 1, float)[:, np.newaxis]
        
        x0 = center_x
        y0 = center_y
        
        return np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))


# ============================================================================
# 2. MODEL ARCHITECTURE
# ============================================================================

class KeypointEstimationModel(nn.Module):
    """Simple keypoint estimation model using ResNet backbone"""
    
    def __init__(self, num_keypoints=17, pretrained=True):
        super(KeypointEstimationModel, self).__init__()
        
        # Backbone: ResNet50
        resnet = resnet50(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # Upsampling layers
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        self.upsample3 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Final prediction layer
        self.output_layer = nn.Conv2d(256, num_keypoints, kernel_size=1)
        
    def forward(self, x):
        # Backbone
        x = self.backbone(x)  # Output: (B, 2048, H/32, W/32)
        
        # Upsample
        x = self.upsample1(x)  # (B, 1024, H/16, W/16)
        x = self.upsample2(x)  # (B, 512, H/8, W/8)
        x = self.upsample3(x)  # (B, 256, H/4, W/4)
        
        # Output heatmaps
        x = self.output_layer(x)  # (B, num_keypoints, H/4, W/4)
        
        return x


# ============================================================================
# 3. LOSS FUNCTION
# ============================================================================

class KeypointMSELoss(nn.Module):
    """MSE loss for keypoint heatmap regression"""
    
    def __init__(self):
        super(KeypointMSELoss, self).__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, pred, target):
        return self.mse(pred, target)


# ============================================================================
# 4. TRAINING FUNCTIONS
# ============================================================================

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


# ============================================================================
# 5. MAIN TRAINING SCRIPT
# ============================================================================

def main():
    # Hyperparameters
    ROOT_DIR = r'C:\Users\beyae\fiftyone\coco-2017'  # Your COCO root directory
    USE_RAW = True  # Set to True to use raw/*.json files, False to use train/labels.json and validation/labels.json
    BATCH_SIZE = 8
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-3
    IMG_SIZE = 256
    HEATMAP_SIZE = 64
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create datasets
    print('Loading datasets...')
    train_dataset = COCOKeypointDataset(
        ROOT_DIR, 
        split='train',
        img_size=IMG_SIZE,
        heatmap_size=HEATMAP_SIZE,
        use_raw=USE_RAW
    )
    
    val_dataset = COCOKeypointDataset(
        ROOT_DIR,
        split='validation', 
        img_size=IMG_SIZE,
        heatmap_size=HEATMAP_SIZE,
        use_raw=USE_RAW
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )
    
    # Initialize model
    print('Initializing model...')
    model = KeypointEstimationModel(num_keypoints=17, pretrained=True)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = KeypointMSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Training loop
    print('Starting training...')
    best_val_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        print(f'\nEpoch {epoch+1}/{NUM_EPOCHS}')
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_keypoint_model.pth')
            print('Saved best model!')
    
    print('Training complete!')


if __name__ == '__main__':
    main()