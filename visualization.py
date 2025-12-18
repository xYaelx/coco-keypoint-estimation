import torch
import cv2
import numpy as np
import os
from config import IMG_SIZE, HEATMAP_SIZE, NUM_KEYPOINTS


COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9),
    (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13),
    (13, 15), (12, 14), (14, 16)
]

KEYPOINT_NAMES = [
    'Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder',
    'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist',
    'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle'
]


def extract_keypoints(heatmaps):
    """Extract keypoints from heatmaps"""
    if isinstance(heatmaps, torch.Tensor):
        heatmaps = heatmaps.cpu().numpy()
    
    if heatmaps.ndim == 4:
        heatmaps = heatmaps[0]
    
    keypoints = []
    for i in range(heatmaps.shape[0]):
        hm = heatmaps[i]
        conf = np.max(hm)
        if conf > 0.1:
            y, x = np.unravel_index(np.argmax(hm), hm.shape)
            x = int(x * IMG_SIZE / HEATMAP_SIZE)
            y = int(y * IMG_SIZE / HEATMAP_SIZE)
            keypoints.append((x, y))
        else:
            keypoints.append(None)
    return keypoints


def draw_keypoints(image, keypoints, color=(0, 255, 0)):
    """Draw keypoints and skeleton on image"""
    img = image.copy()
    
    # Draw skeleton
    for start, end in COCO_SKELETON:
        if keypoints[start] and keypoints[end]:
            pt1 = keypoints[start]
            pt2 = keypoints[end]
            cv2.line(img, pt1, pt2, color, 2)
    
    # Draw keypoints
    for kpt in keypoints:
        if kpt:
            cv2.circle(img, kpt, 4, color, -1)
    
    return img


def save_sample(image, pred_hm, gt_hm, filename):
    """Save a single sample visualization"""
    # Normalize image
    if isinstance(image, torch.Tensor):
        img = image.cpu().numpy()
        if img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img = image
    
    # Extract keypoints
    pred_kpts = extract_keypoints(pred_hm)
    gt_kpts = extract_keypoints(gt_hm)
    
    # Draw on separate images
    img_pred = draw_keypoints(img, pred_kpts, color=(0, 255, 0))
    img_gt = draw_keypoints(img, gt_kpts, color=(0, 0, 255))
    
    # Side by side
    result = np.hstack([img_gt, img_pred])
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    cv2.imwrite(filename, result)
    print(f'Saved: {filename}')
