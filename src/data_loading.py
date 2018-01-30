import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets.folder import default_loader


class RetinopathyDataset(Dataset):

    def __init__(self, root_dir, csv_file, transform=None, loader=default_loader):
        self.root_dir = Path(root_dir) if type(root_dir) is str else root_dir
        self.image_names = list(self.root_dir.glob('*.jpeg'))
        self.transform = transform
        if csv_file:
            image_name_set = {i.stem for i in self.image_names}
            labels = pd.read_csv(csv_file)
            self.labels = {image: label for image, label in
                           zip(list(labels.image), list(labels.level))
                           if image in image_name_set}
        else:
            self.labels = None
        self.loader = loader

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_path = self.image_names[index]
        image = self.loader(image_path)
        if self.transform:
            image = self.transform(image)

        if self.labels:
            label = self.labels[image_path.stem]
            return image, label
        else:
            return image, image_path.stem


class LabelBalancer:
    def __init__(self, y):
        self.y = np.asarray(list(y))

    def _try_frac(self, m, n, pn):
        # Return (a, b) s.t. a <= m, b <= n
        # and b / a is as close to pn as possible
        r = int(round(float(pn * m) / (1.0 - pn)))
        s = int(round(float((1.0 - pn) * n) / pn))
        return (m, r) if r <= n else ((s, n) if s <= m else (m, n))

    def _get_counts(self, nneg, npos, frac_pos):
        if frac_pos > 0.5:
            return self._try_frac(nneg, npos, frac_pos)
        else:
            return self._try_frac(npos, nneg, 1.0 - frac_pos)[::-1]

    def _get_row_counts(self, num_classes):
        row_pos = []
        row_n = []
        for i in range(num_classes):
            curr_column_pos = np.where(self.y == i)[0]
            if len(curr_column_pos) == 0:
                raise ValueError(f"No positive labels for row {i}.")
            row_pos.append(curr_column_pos)
            row_n.append(len(curr_column_pos))
            logging.info(f'Found {len(curr_column_pos)} samples for category {i}')
        return row_pos, row_n

    def rebalance_categorical_train_idxs_pos_neg(self, rebalance=0.5, num_classes=5, rand_state=None):
        """Get training indices based on @y
            @rebalance: bool or fraction of positive examples desired
                        If True, default fraction is 0.5. If False no balancing.
        """
        rs = np.random if rand_state is None else rand_state
        row_pos, row_n = self._get_row_counts(num_classes)
        n_neg = row_n[0]
        n_pos = sum(row_n[1:])
        p = 0.5 if rebalance is True else rebalance
        n_neg, n_pos = self._get_counts(n_neg, n_pos, p)
        row_pos[0] = rs.choice(row_pos[0], size=n_neg, replace=False)
        for i in range(1, len(row_pos)):
            row_pos[i] = rs.choice(row_pos[i],
                                   size=min(int(n_pos / (num_classes - 1)), row_n[i]),
                                   replace=False)
        idxs = np.concatenate(row_pos)
        rs.shuffle(idxs)
        return list(idxs)

    def rebalance_categorical_train_idxs_evenly(self, num_classes=5, rand_state=None):
        """Get training indices based on @y
            @rebalance: bool or fraction of positive examples desired
                        If True, default fraction is 0.5. If False no balancing.
        """
        rs = np.random if rand_state is None else rand_state
        row_pos, row_n = self._get_row_counts(num_classes)
        min_n = min(row_n)
        for i in range(num_classes):
            row_pos[i] = rs.choice(row_pos[i], size=min_n, replace=False)
        idxs = np.concatenate(row_pos)
        rs.shuffle(idxs)
        return idxs

    def rebalance_categorical_train_idxs_almost_evenly(self, num_classes=5, rand_state=None):
        """Get training indices based on @y
            @rebalance: bool or fraction of positive examples desired
                        If True, default fraction is 0.5. If False no balancing.
        """
        rs = np.random if rand_state is None else rand_state
        row_pos, row_n = self._get_row_counts(num_classes)
        min_n = min(row_n)
        for i in range(num_classes):
            row_pos[i] = rs.choice(row_pos[i], size=min(row_n[i], int(min_n * 1.5)), replace=False)
        idxs = np.concatenate(row_pos)
        rs.shuffle(idxs)
        return idxs


# customization of https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb
def get_train_valid_loader(data_dir,
                           label_path,
                           batch_size,
                           train_transforms,
                           valid_transforms,
                           random_seed,
                           rebalance_strategy,
                           valid_size=0.1,
                           shuffle=True,
                           num_workers=4,
                           pin_memory=False):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the CIFAR-10 dataset. A sample
    9x9 grid of the images can be optionally displayed.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - show_sample: plot 9x9 sample grid of the dataset.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    train_dataset = RetinopathyDataset(data_dir, label_path,
                                       train_transforms)

    valid_dataset = RetinopathyDataset(data_dir, label_path,
                                       valid_transforms)

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle == True:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    if rebalance_strategy in {'almost_even', 'even', 'posneg'}:
        label_balancer = LabelBalancer(train_dataset.labels.values())
        logging.info(f'Train samples before rebalancing: {len(train_idx)}')
        if rebalance_strategy == 'even':
            train_idx = label_balancer.rebalance_categorical_train_idxs_evenly()
        elif rebalance_strategy == 'almost_even':
            train_idx = label_balancer.rebalance_categorical_train_idxs_almost_evenly()
        else:
            train_idx = label_balancer.rebalance_categorical_train_idxs_pos_neg()
        logging.info(f'Train samples after rebalancing: {len(train_idx)}')
    elif rebalance_strategy is not None:
        logging.info('Could not recognize rebalance_strategy. Not rebalancing')

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, sampler=train_sampler,
                                               num_workers=num_workers, pin_memory=pin_memory)

    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=batch_size, sampler=valid_sampler,
                                               num_workers=num_workers, pin_memory=pin_memory)

    return (train_loader, valid_loader)


def get_test_loader(data_dir,
                    batch_size,
                    transforms,
                    shuffle=False,
                    num_workers=4,
                    pin_memory=False):
    """
    Utility function for loading and returning a multi-process
    test iterator over the CIFAR-10 dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - shuffle: whether to shuffle the dataset after every epoch.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - data_loader: test set iterator.
    """

    dataset = RetinopathyDataset(data_dir, None, transforms)

    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              pin_memory=pin_memory)

    return data_loader
