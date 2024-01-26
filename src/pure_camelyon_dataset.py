import os
import h5py
import torch
from torch.utils.data import Dataset
import numpy as np

class PureCamelyonDataset(Dataset):
    def __init__(self, hdf5_path, wsis, transform=None):
        """
        Initialize the dataset with patches from specified Whole Slide Images (WSIs).

        Args:
            hdf5_path (str): Path to the HDF5 file containing patch data.
            wsis (list): List of WSI names from which patches will be used.
            transform (callable, optional): Transform to be applied on each patch sample. Defaults to None.
        """
        self.hdf5_path = hdf5_path
        self.wsis = wsis
        self.transform = transform

        # Validate and load patch keys
        self.positive_keys, self.negative_keys = self._load_patch_keys()
        self.index_mapping = self._create_index_mapping()

    def _load_patch_keys(self):
        """
        Load and sample patch keys from the HDF5 file based on data_percent and pos_neg_ratio.
        """
        wsi_positive_keys = {}
        wsi_negative_keys = {}

        with h5py.File(self.hdf5_path, 'r') as file:
            for wsi in self.wsis:
                if wsi not in file:
                    raise ValueError(f"WSI {wsi} not found in HDF5 file")
                
                wsi_positive_keys[wsi] = list(file[wsi]['positives']['patches'].keys())
                wsi_negative_keys[wsi] = list(file[wsi]['negatives']['patches'].keys())

        return wsi_positive_keys, wsi_negative_keys
    
    def _create_index_mapping(self):
        """
        Create a mapping from a linear index to (group_name, patch_key).
        """
        mapping = [(wsi, 'positives', key) for wsi in self.wsis for key in self.positive_keys[wsi]]
        mapping += [(wsi, 'negatives', key) for wsi in self.wsis for key in self.negative_keys[wsi]]
        
        # Shuffle the mapping to mix positive and negative patches
        np.random.shuffle(mapping)
        return mapping

    def __len__(self):
        return len(self.index_mapping)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.index_mapping):
            raise IndexError("Index out of range")

        wsi_name, group_name, patch_key = self.index_mapping[idx]
        label_key = patch_key.replace('patch', 'label')

        with h5py.File(self.hdf5_path, 'r') as file:
            patch = file[wsi_name][group_name]['patches'][patch_key][()]
            label = file[wsi_name][group_name]['labels'][label_key][()]
            label = torch.tensor(label, dtype=torch.long)
            if self.transform:
                patch = self.transform(patch)

            metadata = {
                'original_file': file[wsi_name][group_name]['patches'][patch_key].attrs['original_file'],
                'original_patch_origin': file[wsi_name][group_name]['patches'][patch_key].attrs['original_patch_origin'],
            }
        return patch, label, metadata
    