import torch
from torch.utils.data import Dataset
import h5py
import os

class CamelyonDataset(Dataset):
    def __init__(self, hdf5_paths, output_level, patch_size, transform=None):
        """
        Initialize the dataset.
        :param hdf5_paths: List of paths to the HDF5 files.
        :param output_level: The level of the WSI from which patches were extracted.
        :param patch_size: Size of the patches.
        :param transform: Optional transform to be applied on a sample.
        """
        self.hdf5_paths = hdf5_paths
        self.output_level = output_level
        self.patch_size = patch_size
        self.transform = transform
        self.index_mapping = self._create_index_mapping()

        # Check the structure and existence of the HDF5 files
        for hdf5_path in self.hdf5_paths:
            if not os.path.exists(hdf5_path):
                raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

    def _create_index_mapping(self):
        """
        Create a mapping from a linear index to (file_index, patch_index).
        This helps in fetching the correct patch from the correct file.
        """
        mapping = []
        for file_index, hdf5_path in enumerate(self.hdf5_paths):
            wsi_name = os.path.splitext(os.path.basename(hdf5_path))[0]
            try:
                with h5py.File(hdf5_path, 'r') as file:
                    group_path = f"{wsi_name}/Level_{self.output_level}/Patch_Size_{self.patch_size[0]}"
                    num_patches = file[group_path].attrs['total_patches']
                    for patch_index in range(num_patches):
                        mapping.append((file_index, patch_index))
            except (FileNotFoundError, KeyError) as e:
                raise ValueError(f"Error accessing HDF5 file: {hdf5_path}. {str(e)}")

        return mapping

    def __len__(self):
        """
        Get the total number of patches in the dataset.
        :return: The length of the dataset.
        """
        return len(self.index_mapping)

    def __getitem__(self, idx):
        """
        Get a specific patch and its corresponding label from the dataset.
        :param idx: The index of the patch to retrieve.
        :return: The patch and its label.
        """
        if idx < 0 or idx >= len(self.index_mapping):
            raise IndexError("Index out of range")

        file_index, patch_index = self.index_mapping[idx]
        hdf5_path = self.hdf5_paths[file_index]
        wsi_name = os.path.splitext(os.path.basename(hdf5_path))[0]

        try:
            with h5py.File(hdf5_path, 'r') as file:
                group_path = f"{wsi_name}/Level_{self.output_level}/Patch_Size_{self.patch_size[0]}"
                patch_group = file[f"{group_path}/patches"]
                label_group = file[f"{group_path}/labels"]

                patch_name = f"patch_{patch_index:05d}"
                label_name = f"label_{patch_index:05d}"

                patch = patch_group[patch_name][()]
                label = label_group[label_name][()]

                label = torch.tensor(label, dtype=torch.long)

                if self.transform:
                    patch = self.transform(patch)
                print(file_index, patch_index, patch_group[patch_name].atts['patch_origin'])

        except (FileNotFoundError, KeyError) as e:
            raise ValueError(f"Error accessing HDF5 file: {hdf5_path}. {str(e)}")

        return patch, label
    