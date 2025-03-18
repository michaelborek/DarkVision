from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from PIL import Image
import os

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, folder, image_transform=None, label_transform=None):
        self.image_folder = folder+'images'
        self.label_folder = folder+'labels'
        self.image_files = sorted(os.listdir(self.image_folder))
        self.label_files = sorted(os.listdir(self.label_folder))
        self.image_transform = image_transform
        self.label_transform = label_transform

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        label_file = self.label_files[idx]  # Use the same index
        
        image_path = os.path.join(self.image_folder, image_file)
        label_path = os.path.join(self.label_folder, label_file)
        
        # Process the image
        image = Image.open(image_path).convert('RGB')
        if self.image_transform:
            image = self.image_transform(image)
        else:
            image = np.array(image) / 255.0
            image = torch.tensor(image, dtype=torch.float).permute(2, 0, 1)
        
        # Process the label file (assuming it contains text annotations)
        with open(label_path, 'r') as f:
            label_line = f.readline().strip()
            label_values = [float(x) for x in label_line.split()]
            label = torch.tensor(label_values, dtype=torch.float)
        
        return image, label

    def __len__(self):
        return len(self.image_files)


def DataLoaderFunc(file_name):
    train_dataset = CustomDataset(file_name)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    return train_loader
