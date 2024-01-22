import os
import h5py
import torch
from torch.utils.data import Dataset
import numpy as np

class PureCamelyonDataset(Dataset):
    def __init__(self, hdf5_path, data_percent=1.0, pos_neg_ratio=None, transform=None):
        """
        Initialize the dataset.
        :param hdf5_path: Path to the HDF5 file.
        :param data_percent: Portion of the dataset to use (between 0 and 1).
        :param pos_neg_ratio: Desired ratio of positive to negative patches (e.g., 0.5 for equal amounts).
        :param transform: Optional transform to be applied on a sample.
        """
        self.hdf5_path = hdf5_path
        self.data_percent = data_percent
        self.pos_neg_ratio = pos_neg_ratio
        self.transform = transform

        # Validate and load patch keys
        self.positive_keys, self.negative_keys = self._load_patch_keys()
        self.index_mapping = self._create_index_mapping()

    def _load_patch_keys(self):
        """
        Load and sample patch keys from the HDF5 file based on data_percent and pos_neg_ratio.
        """
        with h5py.File(self.hdf5_path, 'r') as file:
            all_positive_keys = list(file['positives']['patches'].keys())
            all_negative_keys = list(file['negatives']['patches'].keys())

            # Determine the number of positive patches to use
            num_positive_to_use = len(all_positive_keys) if self.data_percent == 1.0 else int(len(all_positive_keys) * self.data_percent)

            # Determine the number of negative patches to use based on pos_neg_ratio
            if self.pos_neg_ratio is not None:
                # Calculate the total number of patches needed to achieve the desired ratio
                neg_patches_needed = int(num_positive_to_use / self.pos_neg_ratio) - num_positive_to_use
                num_negative_to_use = min(neg_patches_needed, len(all_negative_keys))  # Limit to available negatives
            else:
                num_negative_to_use = len(all_negative_keys) if self.data_percent == 1.0 else int(len(all_negative_keys) * self.data_percent)

            # Sample patches
            if self.data_percent < 1.0:
                positive_keys = np.random.choice(all_positive_keys, num_positive_to_use, replace=False)
                negative_keys = np.random.choice(all_negative_keys, num_negative_to_use, replace=False)
                return positive_keys.tolist(), negative_keys.tolist()
            else:
                return all_positive_keys, all_negative_keys
    def _create_index_mapping(self):
        """
        Create a mapping from a linear index to (group_name, patch_key).
        """
        mapping = []
        for key in self.positive_keys:
            mapping.append(('positives', key))
        for key in self.negative_keys:
            mapping.append(('negatives', key))

        # Shuffle the mapping to mix positive and negative patches
        #np.random.shuffle(mapping)
        return mapping

    def __len__(self):
        return len(self.index_mapping)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.index_mapping):
            raise IndexError("Index out of range")

        group_name, patch_key = self.index_mapping[idx]
        label_key = patch_key.replace('patch', 'label')

        with h5py.File(self.hdf5_path, 'r') as file:
            patch = file[group_name]['patches'][patch_key][()]
            label = file[group_name]['labels'][label_key][()]
            label = torch.tensor(label, dtype=torch.long)
            origin = file[group_name]['patches'][patch_key].attrs['orginal_patch_name']
            if self.transform:
                patch = self.transform(patch)
            print(patch_key, label_key, origin)
        return patch, label
    