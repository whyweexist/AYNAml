import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from typing import Tuple, Dict, List
import random

class PolygonColorDataset(Dataset):
    def __init__(self, data_dir: str, split: str = 'training', 
                 image_size: int = 256, augment: bool = True):
        """
        Dataset for polygon coloring task
        
        Args:
            data_dir: Path to dataset directory
            split: 'training' or 'validation'
            image_size: Size to resize images to
            augment: Whether to apply data augmentation
        """
        self.data_dir = data_dir
        self.split = split
        self.image_size = image_size
        self.augment = augment and (split == 'training')
        
        # Paths
        self.split_dir = os.path.join(data_dir, split)
        self.inputs_dir = os.path.join(self.split_dir, 'inputs')
        self.outputs_dir = os.path.join(self.split_dir, 'outputs')
        self.json_path = os.path.join(self.split_dir, 'data.json')
        
        # Load data mapping
        with open(self.json_path, 'r') as f:
            self.data = json.load(f)
        
        # Color mapping
        self.color_mapper = ColorMapper()
        
        # Transforms
        self.setup_transforms()
        
    def setup_transforms(self):
        """Setup image transforms"""
        # Base transforms
        base_transforms = [
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ]
        
        # Augmentation transforms for training
        if self.augment:
            aug_transforms = [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                transforms.ToTensor(),
            ]
            self.input_transform = transforms.Compose(aug_transforms)
            self.output_transform = transforms.Compose(aug_transforms)
        else:
            self.input_transform = transforms.Compose(base_transforms)
            self.output_transform = transforms.Compose(base_transforms)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load images
        input_path = os.path.join(self.inputs_dir, item['input_image'])
        output_path = os.path.join(self.outputs_dir, item['output_image'])
        
        input_image = Image.open(input_path).convert('RGB')
        target_image = Image.open(output_path).convert('RGB')
        
        # Apply same augmentation to both input and target if training
        if self.augment:
            # Use same random seed for both transforms to ensure consistency
            seed = random.randint(0, 2**32)
            
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            input_tensor = self.input_transform(input_image)
            
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            target_tensor = self.output_transform(target_image)
        else:
            input_tensor = self.input_transform(input_image)
            target_tensor = self.output_transform(target_image)
        
        # Get color ID
        color_name = item['color'].lower()
        color_id = self.color_mapper.color_name_to_id(color_name)
        
        return {
            'input': input_tensor,
            'target': target_tensor,
            'color_id': torch.tensor(color_id, dtype=torch.long),
            'color_name': color_name,
            'input_image_name': item['input_image'],
            'output_image_name': item['output_image']
        }

class ColorMapper:
    def __init__(self):
        self.color_to_id = {
            'red': 0, 'blue': 1, 'green': 2, 'yellow': 3, 'purple': 4,
            'orange': 5, 'pink': 6, 'brown': 7, 'black': 8, 'white': 9
        }
        self.id_to_color = {v: k for k, v in self.color_to_id.items()}
    
    def color_name_to_id(self, color_name):
        return self.color_to_id.get(color_name.lower(), 0)
    
    def id_to_color_name(self, color_id):
        return self.id_to_color.get(color_id, 'red')

def create_dataloaders(data_dir: str, batch_size: int = 16, 
                      image_size: int = 256, num_workers: int = 4):
    """Create training and validation dataloaders"""
    
    train_dataset = PolygonColorDataset(
        data_dir=data_dir,
        split='training',
        image_size=image_size,
        augment=True
    )
    
    val_dataset = PolygonColorDataset(
        data_dir=data_dir,
        split='validation',
        image_size=image_size,
        augment=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

# Synthetic data generation (optional enhancement)
def generate_synthetic_polygon(shape_type: str, color: str, size: int = 256):
    """Generate synthetic polygon data for augmentation"""
    import cv2
    
    # Create blank image
    img = np.zeros((size, size, 3), dtype=np.uint8)
    
    # Color mapping
    color_map = {
        'red': (255, 0, 0), 'blue': (0, 0, 255), 'green': (0, 255, 0),
        'yellow': (255, 255, 0), 'purple': (128, 0, 128), 'orange': (255, 165, 0),
        'pink': (255, 192, 203), 'brown': (165, 42, 42), 'black': (0, 0, 0), 'white': (255, 255, 255)
    }
    
    color_rgb = color_map.get(color, (255, 0, 0))
    center = (size // 2, size // 2)
    
    if shape_type == 'triangle':
        pts = np.array([[center[0], center[1] - 80],
                       [center[0] - 70, center[1] + 60],
                       [center[0] + 70, center[1] + 60]], np.int32)
        cv2.fillPoly(img, [pts], color_rgb)
    elif shape_type == 'square':
        cv2.rectangle(img, (center[0] - 60, center[1] - 60), 
                     (center[0] + 60, center[1] + 60), color_rgb, -1)
    elif shape_type == 'circle':
        cv2.circle(img, center, 70, color_rgb, -1)
    elif shape_type == 'hexagon':
        pts = []
        for i in range(6):
            angle = i * np.pi / 3
            x = int(center[0] + 70 * np.cos(angle))
            y = int(center[1] + 70 * np.sin(angle))
            pts.append([x, y])
        pts = np.array(pts, np.int32)
        cv2.fillPoly(img, [pts], color_rgb)
    
    return img

if __name__ == "__main__":
    # Test the dataset
    data_dir = "./dataset"  # Update this path
    
    # This would work if dataset is available
    try:
        train_loader, val_loader = create_dataloaders(data_dir, batch_size=4)
        
        # Test loading a batch
        for batch in train_loader:
            print(f"Input shape: {batch['input'].shape}")
            print(f"Target shape: {batch['target'].shape}")
            print(f"Color IDs: {batch['color_id']}")
            print(f"Color names: {batch['color_name']}")
            break
            
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        
    except Exception as e:
        print(f"Dataset not found or error loading: {e}")
        print("Please ensure dataset is downloaded and extracted to the correct path")
        
        # Demo synthetic data generation
        print("\nGenerating synthetic polygon example:")
        synthetic_img = generate_synthetic_polygon('triangle', 'red')
        print(f"Generated synthetic polygon shape: {synthetic_img.shape}")
