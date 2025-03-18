from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from PIL import Image
import pandas as pd

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
        # Get the row corresponding to the idx
        row = self.data.iloc[idx]
        image_path = row['filepath']
        label = row['label']

        # Load and process the image
        image = Image.open(image_path).convert('RGB')
        if self.image_transform:
            image = self.image_transform(image)
        else:
            image = np.array(image) / 255.0
            image = torch.tensor(image, dtype=torch.float).permute(2, 0, 1)

        # Process the label: ensure it is an integer and then a tensor of type long
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
