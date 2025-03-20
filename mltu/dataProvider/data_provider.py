import os
import numpy as np
from typing import List, Tuple

class DataProvider:
    def __init__(self, data_path: str, labels_path: str = None, batch_size: int = 32, shuffle: bool = True):
        self.data_path = data_path
        self.labels_path = labels_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.data = []
        self.labels = []
        self.load_data()
        
    def load_data(self):
        """Load data from the specified paths"""
        if os.path.exists(self.data_path):
            self.data = [os.path.join(self.data_path, f) for f in os.listdir(self.data_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
            
        if self.labels_path and os.path.exists(self.labels_path):
            with open(self.labels_path, 'r', encoding='utf-8') as f:
                self.labels = f.readlines()
                
        if len(self.data) != len(self.labels) and self.labels_path:
            raise ValueError("Number of images does not match number of labels")
            
    def __len__(self) -> int:
        """Return the number of samples in the dataset"""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[str, str]:
        """Get a single sample from the dataset"""
        image_path = self.data[idx]
        label = self.labels[idx].strip() if self.labels else ""
        return image_path, label
    
    def get_batch(self, batch_idx: int) -> Tuple[List[str], List[str]]:
        """Get a batch of samples"""
        start_idx = batch_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self))
        
        batch_images = []
        batch_labels = []
        
        for idx in range(start_idx, end_idx):
            image_path, label = self[idx]
            batch_images.append(image_path)
            batch_labels.append(label)
            
        return batch_images, batch_labels
    
    def on_epoch_end(self):
        """Called at the end of every epoch"""
        if self.shuffle:
            indices = np.arange(len(self))
            np.random.shuffle(indices)
            self.data = [self.data[i] for i in indices]
            if self.labels:
                self.labels = [self.labels[i] for i in indices] 