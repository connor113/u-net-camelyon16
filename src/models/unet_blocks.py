import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    # Initialize the DoubleConv block
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels

        # Define the double conv block with two conv layers
        # Each conv layer is a sequence of Conv2d, BatchNorm2d, ReLU
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    # Define the forward pass
    def forward(self, x):
        return self.double_conv(x)


class DownBlock(nn.Module):
    """Downscaling with maxpool then double conv"""

    # Initialize the DownBlock
    def __init__(self, in_channels, out_channels):
        # Initialize the parent class
        super(DownBlock, self).__init__()

        # Define the maxpool-conv block
        self.maxpool_conv = nn.Sequential(

            # Max Pooling layer with kernel size 2
            # This will downscale the spacial dimentions by a factor of 2
            nn.MaxPool2d(2),
            # Double Convolution block
            DoubleConv(in_channels, out_channels)
        )
        
    # Define the forward pass
    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upscaling then double conv"""

    # Initialize the UpBlock
    def __init__(self, in_channels, out_channels, bilinear=True):
        # Initialize the parent class
        super(UpBlock, self).__init__()

        # Define the upsampling layer
        if bilinear:
            # If using bilinear upsampling
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            # If using transposed convolution
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    # Define the forward pass
    def forward(self, x1, x2):
        # Upsample the input
        x1 = self.up(x1)

        # Pad the input if the shapes are not the same
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # Pad the input tensor x1
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # Concatenate the tensors along the channels axis
        x = torch.cat([x2, x1], dim=1)

        # Return the double conv block
        return self.conv(x)


class OutConv(nn.Module):
    """Output layer 1x1 convolution to get pixel predictions"""
    
    # Initialize the OutConv layer
    def __init__(self, in_channels, out_channels):
        # Initialize the parent class
        super(OutConv, self).__init__()

        # Define the 1x1 convolution layer
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    # Define the forward pass
    def forward(self, x):
        return self.conv(x)