"""Assembles the U-Net model from the blocks defined in unet_blocks.py"""
import torch
import torch.nn as nn
from .unet_blocks import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, use_checkpoint=False):
        super(UNet, self).__init__()

        # Initialize variables
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.use_checkpoint = use_checkpoint

        # Define the encoder part of the U-Net model
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = DownBlock(64, 128)
        self.down2 = DownBlock(128, 256)
        self.down3 = DownBlock(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = DownBlock(512, 1024 // factor)

        # Define the decoder part of the U-Net model
        self.up1 = UpBlock(1024, 512 // factor, bilinear)
        self.up2 = UpBlock(512, 256 // factor, bilinear)
        self.up3 = UpBlock(256, 128 // factor, bilinear)
        self.up4 = UpBlock(128, 64, bilinear)

        # Define the output layer
        self.outc = OutConv(64, n_classes)

        # Define whether to use checkpointing
        if self.use_checkpoint:
            self.enable_checkpointing()

    def forward(self, x):
        # Define the forward pass through the U-Net model
        # Encoder part
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder part with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Output layer
        logits = self.outc(x)
        return logits
    
    def enable_checkpointing(self):
        # Enable checkpointing
        self.inc = torch.utils.checkpoint.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint.checkpoint(self.outc)
