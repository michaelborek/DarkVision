from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from PIL import Image
import pandas as pd
import torch.nn as nn
from loss import FocalLoss, MSELoss  

class CustomDataset(Dataset):
    def __init__(self, csv_file, image_transform=None, label_transform=None):
        """
        Args:
            csv_file (str): Path to the CSV file with image paths and labels.
            image_transform (callable, optional): Optional transform to be applied to images.
            label_transform (callable, optional): Optional transform to be applied to labels.
        """
        self.data = pd.read_csv(csv_file)
        self.image_transform = image_transform
        self.label_transform = label_transform

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = row['filepath']
        label = row['label']

        image = Image.open(image_path).convert('RGB')
        if self.image_transform:
            image = self.image_transform(image)
        else:
            image = np.array(image) / 255.0
            image = torch.tensor(image, dtype=torch.float).permute(2, 0, 1)

        label = int(label)
        label = torch.tensor(label, dtype=torch.long)
        if self.label_transform:
            label = self.label_transform(label)
        
        return image, label

    def __len__(self):
        return len(self.data)

def DataLoaderFunc(csv_file, batch_size=64, shuffle=True, image_transform=None, label_transform=None):
    dataset = CustomDataset(csv_file, image_transform=image_transform, label_transform=label_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader

def get_loss_function(loss_type='focal', **kwargs):
    """
    Select and return a loss function based on the loss_type argument.

    Args:
        loss_type (str): Type of loss function to use. Supported values are:
                         'focal' for FocalLoss, 'mse' for MSELoss, and 
                         'cross_entropy' (or 'ce') for PyTorch's built-in CrossEntropyLoss.
        **kwargs: Additional keyword arguments passed to the loss function's constructor.
        
    Returns:
        torch.nn.Module: An instance of the selected loss function.
    
    Raises:
        ValueError: If an unsupported loss_type is provided.
    """
    loss_type = loss_type.lower()
    if loss_type == 'focal':
        return FocalLoss(**kwargs)
    elif loss_type == 'mse':
        return MSELoss(**kwargs)
    elif loss_type in ['cross_entropy', 'ce']:
        return nn.CrossEntropyLoss(**kwargs)
    else:
        raise ValueError(f"Unsupported loss function type: {loss_type}")
