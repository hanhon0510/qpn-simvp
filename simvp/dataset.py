from torch.utils.data import Dataset
import torch
from tifffile import imread
import os
import numpy as np

class TIFDataset(Dataset):
    def __init__(self, root_dir):
        print(f"Looking in root_dir: {root_dir}")
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"The provided root directory does not exist: {root_dir}")
        self.file_paths = []
        for subdir, dirs, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith('.tif'):
                    self.file_paths.append(os.path.join(subdir, file))
        self.file_paths = sorted(self.file_paths)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        image = imread(file_path)
        
        # Assuming the image is single-channel; add channel dimension if it's not present
        if image.ndim == 2:
            image = np.expand_dims(image, axis=0)  # Add channel dimension
        elif image.ndim == 3:
            image = np.transpose(image, (2, 0, 1))  # Move channel to first dimension if it's last

        # Handle NoData values; consider replacing them with 0 or another suitable value
        image[image == -9999] = 0  # Replace NoData value with 0 or a suitable placeholder
        image = np.clip(image, 0, None)  # Assuming you want to clip negative values

        # Normalize image
        image = torch.tensor(image, dtype=torch.float32)
        image /= image.max()  # Simple normalization by the max value

        return image

