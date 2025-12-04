import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
from pycocotools.coco import COCO
import os


class COCOKeypointDataset(Dataset):
    """COCO Keypoint Dataset for person keypoint detection"""
    
    def __init__(self, root_dir, split='train', img_size=256, heatmap_size=64, use_raw=True, max_samples=None):
        """
        Args:
            root_dir: Root directory containing COCO data (e.g., C:\\Users\\beyae\\fiftyone\\coco-2017)
            split: 'train' or 'validation'
            img_size: Input image size
            heatmap_size: Output heatmap resolution
            use_raw: If True, use annotation files from raw folder, otherwise use labels.json
            max_samples: If not None, limit the number of samples in the dataset
        """
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.heatmap_size = heatmap_size
        self.sigma = 2  # Gaussian sigma for heatmap generation
        
        # COCO has 17 keypoints for each person
        self.num_keypoints = 17
        
        # raw is the original COOCO annotations while train/labels.json and validation/labels.json are custom by FiftyOne
        if split == 'train':
            self.img_dir = os.path.join(root_dir, 'train', 'data')
            if use_raw:
                self.ann_file = os.path.join(root_dir, 'raw', 'person_keypoints_train2017.json')
            else:
                self.ann_file = os.path.join(root_dir, 'train', 'labels.json')
        else:  
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
        
        # Limit dataset size if max_samples is specified
        if max_samples is not None:
            self.valid_img_ids = self.valid_img_ids[:max_samples]
        
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
        
        # Load annotations
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
