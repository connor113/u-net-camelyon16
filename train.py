import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import os
from camelyon_dataset import CamelyonDataset
from models.unet_model import UNet
from torch.utils.data import DataLoader

def main():
    shit_here



def prepare_data(train_hdf5_paths, val_hdf5_paths, output_level, patch_size, batch_size, num_workers):
    # Create the CamelyonDataset instances for training and validation
    train_dataset = CamelyonDataset(hdf5_paths=train_hdf5_paths,
                                    output_level=output_level,
                                    patch_size=patch_size)

    val_dataset = CamelyonDataset(hdf5_paths=val_hdf5_paths,
                                  output_level=output_level,
                                  patch_size=patch_size)

    # Create DataLoaders for training and validation datasets
    train_loader = DataLoader(train_dataset, 
                              batch_size=batch_size, 
                              shuffle=True, 
                              num_workers=num_workers)

    val_loader = DataLoader(val_dataset, 
                            batch_size=batch_size, 
                            shuffle=False, 
                            num_workers=num_workers)

    return train_loader, val_loader