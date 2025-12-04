"""
Keypoint Estimation Model Architecture
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50


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
