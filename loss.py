import torch
import torch.nn as nn


class KeypointMSELoss(nn.Module):
    """MSE loss for keypoint heatmap regression"""
    
    def __init__(self):
        super(KeypointMSELoss, self).__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, pred, target):
        return self.mse(pred, target)
