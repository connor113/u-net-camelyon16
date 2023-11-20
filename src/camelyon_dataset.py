import torch
from torch.utils.data import Dataset
import h5py
import os

class CamelyonDataset(Dataset):
    def __init__(self, root_dir, train=True, transforms=None):
        """
        Args:
            root_dir (string): Directory with all the H5 files.
            train (bool, optional): If True, use training set. Use test set otherwise.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = os.path.join(root_dir, 'train' if train else 'test')
        self.transforms = transforms
        self.sample_counts = self.count_samples_in_hdf5()

    def count_samples_in_hdf5(self):
        """
        Count the number of samples in HDF5 files within the root directory of the dataset.

        Returns:
            int: The total number of samples across all HDF5 files.
        """
        total_samples = 0
        # Iterate over all HDF5 files in the root directory
        for hdf5_filename in os.listdir(self.root_dir):
            hdf5_filepath = os.path.join(self.root_dir, hdf5_filename)
            if hdf5_filepath.endswith('.h5'):
                with h5py.File(hdf5_filepath, 'r') as f:
                    # Assuming the main group is named after the slide without the '.h5' extension
                    slide_group_name = os.path.splitext(hdf5_filename)[0]
                    if slide_group_name in f:
                        level_group = f[slide_group_name].get(f"Level_0")
                        if level_group:
                            patch_group = level_group.get("patches")
                            if patch_group:
                                total_samples += len(patch_group)
        return total_samples

    def __len__(self):
        return self.sample_counts
    
    