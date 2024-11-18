from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import struct
import gzip

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        def parse_minist(image_filesname, label_filename):
            with gzip.open(image_filesname, 'rb') as f:
                # read the date in the big endian order by `unpack``
                magic_number, num_images, num_rows, num_cols = struct.unpack('>IIII', f.read(16))
                image_origin = np.frombuffer(f.read(), dtype=np.uint8) # read an unsigned int byte each time
                image_norm = image_origin.reshape(num_images, num_rows, num_cols, 1).astype(np.float32)
                image_norm /= 255.0

            with gzip.open(label_filename, 'rb') as f:
                magic_number, num_labels = struct.unpack('>II', f.read(8))
                labels = np.frombuffer(f.read(), dtype=np.uint8)

            return image_norm, labels
        
        self.images, self.labels = parse_minist(image_filename, label_filename)
        self.transforms = transforms
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        images = self.images[index]
        labels = self.labels[index]
        images = self.apply_transforms(images)
        return images, labels
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        assert len(self.images) == len(self.labels)
        return len(self.images)
        ### END YOUR SOLUTION