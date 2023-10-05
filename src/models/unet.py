import torch
import torch.nn as nn

class UNet(nn.Module):
    """
    UNet architecture for image segmentation.
    
    Attributes:
        input_channels (int): Number of input channels (e.g., 3 for RGB images).
        output_channels (int): Number of output channels (e.g., 2 for binary segmentation).
        patch_size (int): Size of the input patches (assumed to be square).
        encoder (nn.Module): Encoder part of the U-Net.
        decoder (nn.Module): Decoder part of the U-Net.
        final_conv (nn.Module): Final 1x1 convolution layer.
    """
    
    def __init__(self, input_channels, output_channels, patch_size):
        """
        Initialize the U-Net model.
        
        Parameters:
            input_channels (int): Number of input channels.
            output_channels (int): Number of output channels.
            patch_size (int): Size of the input patches.
        """
        super(UNet, self).__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.patch_size = patch_size
        
        # Build Encoder
        self.encoder = self.build_encoder()
        
        # Build Decoder
        self.decoder = self.build_decoder()
        
        # Build Final Convolution
        self.final_conv = self.build_final_conv()
    
    def build_encoder(self):
        """
        Build the encoder part of the U-Net.
        
        Returns:
            nn.Module: Encoder layers combined as a sequential module.
        """
        encoder_layers = []
        num_filters = 64
        size = self.patch_size
        input_channels = self.input_channels
        
        # Create encoder layers in a loop
        while size >= 4:
            # Convolutional layer with ReLU activation
            conv = nn.Conv2d(input_channels, num_filters, kernel_size=3, padding=1)
            relu = nn.ReLU(inplace=True)
            
            # Max pooling layer for downsampling
            pool = nn.MaxPool2d(2, stride=2)
            encoder_layers.extend([conv, relu, pool])
            
            # Update variables for the next iteration
            input_channels = num_filters
            num_filters *= 2
            size //= 2
        
        return nn.Sequential(*encoder_layers)
    
    def build_decoder(self):
        """
        Build the decoder part of the U-Net.
        
        Returns:
            nn.Module: Decoder layers combined as a sequential module.
        """
        decoder_layers = []
        # Extract the last encoder layer's output channels
        num_filters = self.encoder[-3].out_channels // 2
        input_channels = self.encoder[-3].out_channels
        
        # Create decoder layers in a loop
        while num_filters >= 64:
            # Transposed convolutional layer for upsampling
            upconv = nn.ConvTranspose2d(input_channels, num_filters, kernel_size=2, stride=2)
            
            # Convolutional layer with ReLU activation
            conv = nn.Conv2d(input_channels, num_filters, kernel_size=3, padding=1)
            relu = nn.ReLU(inplace=True)
            decoder_layers.extend([upconv, conv, relu])
            
            # Update variables for the next iteration
            input_channels = num_filters
            num_filters //= 2
        
        return nn.Sequential(*decoder_layers)
    
    def build_final_conv(self):
        """
        Build the final 1x1 convolution layer.
        
        Returns:
            nn.Module: Final 1x1 convolution layer.
        """
        return nn.Conv2d(self.decoder[-3].out_channels, self.output_channels, kernel_size=1)
    
    def forward(self, x):
        """
        Forward pass for the U-Net model.
        
        Parameters:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.encoder(x)  # Pass through the encoder
        x = self.decoder(x)  # Pass through the decoder
        x = self.final_conv(x)  # Pass through the final 1x1 convolution
        return x

# Test the class
# unet_model = UNet(input_channels=3, output_channels=2, patch_size=256)
# print(unet_model)
