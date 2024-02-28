import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, size):
        super(Block, self).__init__()
        self.conv_param = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.dense_ts = nn.Linear(192, out_channels)
        self.layer_norm = nn.LayerNorm([out_channels, size, size])

    def forward(self, x_img, x_ts):
        # First, transform the input with conv_param to match the out_channels
        x_img_transformed = F.relu(self.conv_param(x_img))
        # Process time step information
        time_parameter = F.relu(self.dense_ts(x_ts)).view(-1, x_img_transformed.size(1), 1, 1)
        # Apply the time-dependent transformation
        x_parameter = x_img_transformed * time_parameter
        # Then, pass the transformed input through conv_out
        x_out = self.conv_out(x_parameter)  # x_parameter matches the expected channels for conv_out
        x_out = x_out + x_parameter  # Incorporate the original transformation
        x_out = F.relu(self.layer_norm(x_out))
        return x_out


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.l_ts = nn.Sequential(
            nn.Linear(1, 192),
            nn.LayerNorm([192]),
            nn.ReLU(),
        )

        # Adjust in_channels and out_channels if necessary
        self.down_x64 = Block(in_channels=3, out_channels=128, size=64)  # New block for 64x64 input
        self.down_x32 = Block(in_channels=128, out_channels=128, size=32)
        self.down_x16 = Block(in_channels=128, out_channels=128, size=16)
        self.down_x8 = Block(in_channels=128, out_channels=128, size=8)
        self.down_x4 = Block(in_channels=128, out_channels=128, size=4)
        
        # Adjust MLP input size based on the new flattening size
        self.mlp = nn.Sequential(
            nn.Linear(128 * 4 * 4 + 192, 128),  # This might stay the same if the downsampling ends with 4x4
            nn.LayerNorm([128]),
            nn.ReLU(),
            
            nn.Linear(128, 128 * 4 * 4),  # Output size adjusted based on upsampling requirements
            nn.LayerNorm([128 * 4 * 4]),
            nn.ReLU(),
        )
        
        # Upsampling blocks
        self.up_x4 = Block(in_channels=128, out_channels=128, size=4)
        self.up_x8 = Block(in_channels=128, out_channels=256, size=8)  # Adjust out_channels for feature concatenation
        self.up_x16 = Block(in_channels=256, out_channels=256, size=16)
        self.up_x32 = Block(in_channels=256, out_channels=256, size=32)
        self.up_x64 = Block(in_channels=256, out_channels=128, size=64)  # New block for upscaling to 64x64
        
        self.cnn_output = nn.Conv2d(in_channels=128, out_channels=3, kernel_size=1, padding=0)
        
        self.opt = torch.optim.Adam(self.parameters(), lr=0.0008)
    
    def forward(self, x, x_ts):
        x_ts_processed = self.l_ts(x_ts)

        # Process through downsampling blocks
        # The first block takes the original input with 3 channels
        x = self.down_x64(x, x_ts_processed)
        x = F.max_pool2d(x, 2)  # Downsample to 32x32

        x = self.down_x32(x, x_ts_processed)
        x = F.max_pool2d(x, 2)  # Downsample to 16x16

        x = self.down_x16(x, x_ts_processed)
        x = F.max_pool2d(x, 2)  # Downsample to 8x8

        x = self.down_x8(x, x_ts_processed)
        x = F.max_pool2d(x, 2)  # Downsample to 4x4

        # Last downsampling block, no further downsampling after this
        x = self.down_x4(x, x_ts_processed)

        # MLP - processes the features from the last downsampling block
        x = x.view(-1, 128 * 4 * 4)  # Flatten the features
        x = torch.cat([x, x_ts_processed], dim=1)  # Concatenate with time step information
        x = self.mlp(x)

        x = x.view(-1, 128, 4, 4)  # Reshape for upsampling

        # Process through upsampling blocks, each step doubles the dimensions
        x = self.up_x4(x, x_ts_processed)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # Upsample to 8x8

        x = self.up_x8(x, x_ts_processed)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # Upsample to 16x16

        x = self.up_x16(x, x_ts_processed)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # Upsample to 32x32

        x = self.up_x32(x, x_ts_processed)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # Upsample to 64x64

        # Final upsampling block to get to the original image size
        x = self.up_x64(x, x_ts_processed)

        # Final convolution to produce the output image
        x = self.cnn_output(x)
        return x

