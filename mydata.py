"""
A wrapper for datasets
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
import pickle
import numpy as np

class MyDataset(Dataset):
    def __init__(self, root, transform=None):
        with open(root,'rb') as f:
            data = pickle.load(f, encoding='bytes')
        self.data = np.array(data['data'])
        self.labels = np.array(data['labels'])
        
        self.root = root
        self.transform = transform
        
    def __getitem__(self, index):
        label = self.labels[index]
        if self.transform is not None:
            img = self.transform(self.data[index]) 
        return img, label
        
    def __len__(self):
        return len(self.data)

