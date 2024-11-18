import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset

# The Dataset use numpy as backend, rather than needle's NDArray. 
# I guess NDArray may be only used in compute graph, not in data processing.
class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        def load_data(base_folder: str, train: bool):
            """
            Loads the CIFAR-10 data from the base folder.
            """
            if train:
                files = [f'data_batch_{i}' for i in range(1, 6)]
            else:
                files = ['test_batch']

            data_list = []
            labels_list = []

            for file_name in files:
                file_path = os.path.join(base_folder, file_name)
                with open(file_path, 'rb') as file:
                    batch = pickle.load(file, encoding='bytes')
                    data_list.append(batch[b'data'])
                    labels_list.append(batch[b'labels'])

            X = np.vstack(data_list).reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
            y = np.hstack(labels_list).astype(np.int64)

            return X, y

        self.transforms = transforms
        self.X, self.y = load_data(base_folder, train)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        img, label = self.X[index], self.y[index]
        if self.transforms:
            img = self.transforms(img)
        return img, label
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return len(self.X)
        ### END YOUR SOLUTION

# Author: Qingzheng Wang