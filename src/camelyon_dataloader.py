import torch
from torch.utils.data import Dataset
import h5py
import os

class CamelyonDataset(Dataset):
    def __init__