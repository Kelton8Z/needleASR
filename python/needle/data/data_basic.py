import numpy as np
from ..autograd import Tensor

from typing import Iterator, Optional, List, Sized, Union, Iterable, Any



class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError
    
    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
        collate_fn=None
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.collate_fn = collate_fn if collate_fn is not None else self.collate_fn
        if not self.shuffle:
            self.ordering = np.array_split(np.arange(len(dataset)), 
                                           range(batch_size, len(dataset), batch_size))

    def __iter__(self):
        ### BEGIN YOUR SOLUTION
        order = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(order)
            self.ordering = np.array_split(order, 
                            range(self.batch_size, len(self.dataset), self.batch_size))
        self.index = 0
        ### END YOUR SOLUTION
        return self
    
    def __len__(self):
        return len(self.dataset) // self.batch_size + 1 # don't drop the last batch

    def __next__(self):
        ### BEGIN YOUR SOLUTION
        if self.index == len(self.ordering):
            raise StopIteration
        data = [self.dataset[idx] for idx in self.ordering[self.index]]
        self.index += 1

        # data may contain multiple kinds of data (images, labels) or only one
        # kind without labels
        #
        # FIX @2024/11/25: transfer np.ndarray to collate_fn, the collate_fn transfer 
        # np.ndarray to Tensor this is for the ASR dataset, which needs to pad features, 
        # but Tensor cannot be getitem or setitem, so we must give the collate_fn the 
        # np.ndarray, so the collate_fn could use first pad then transfer to Tensor.

        return self.collate_fn(data)
        ### END YOUR SOLUTION
    
    # default collate_fn, no collation
    def collate_fn(self, batch):
        return tuple(Tensor(x) for x in batch)

# Author: Qingzheng Wang