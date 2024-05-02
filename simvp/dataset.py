from torch.utils.data import Dataset
import torch
from tifffile import imread
import os
import numpy as np

class TIFDataset(Dataset):
    def __init__(self, root_dir, sequence_length=4):
        self.sequence_length = sequence_length
        self.file_paths = [os.path.join(root_dir, fname) for fname in sorted(os.listdir(root_dir)) if fname.lower().endswith('.tif')]
        
        if len(self.file_paths) < sequence_length:
            raise ValueError(f"Not enough data to form a sequence. The directory needs at least {sequence_length} .tif files.")

    def __len__(self):
        # Adjusted to ensure complete sequences are returned
        return len(self.file_paths) - (self.sequence_length - 1)

    def __getitem__(self, idx):
        # Load a sequence of 4 images
        sequence = [imread(self.file_paths[idx + i]) for i in range(4)]
        # ... [rest of the preprocessing]

        # Convert images to tensors and normalize
        sequence = [torch.tensor(img, dtype=torch.float32) / img.max() for img in sequence]
        
        # Stack into a tensor of shape [sequence_length, channels, height, width]
        sequence_tensor = torch.stack(sequence)
        
        # Input is first 3 images, target is the 4th image
        input_tensor = sequence_tensor[:3]  # shape [3, channels, height, width]
        target_tensor = sequence_tensor[3]  # shape [channels, height, width]

        return input_tensor, target_tensor

