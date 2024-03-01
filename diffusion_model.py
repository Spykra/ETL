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
        x_img_transformed = F.relu(self.conv_param(x_img))
        time_parameter = F.relu(self.dense_ts(x_ts)).view(-1, x_img_transformed.size(1), 1, 1)
        x_parameter = x_img_transformed * time_parameter
        x_out = self.conv_out(x_parameter)
        x_out = x_out + x_parameter
        x_out = F.relu(self.layer_norm(x_out))
        return x_out

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l_ts = nn.Sequential(nn.Linear(1, 192), nn.LayerNorm([192]), nn.ReLU())

        # Simplify block creation using loops
        self.down_blocks = nn.ModuleList([Block(3, 128, 64)] + [Block(128, 128, size) for size in [32, 16, 8, 4]])
        self.up_blocks = nn.ModuleList([Block(128, 128, 4), Block(128, 256, 8), Block(256, 256, 16), Block(256, 256, 32), Block(256, 128, 64)])
        
        self.mlp = nn.Sequential(
            nn.Linear(512 + 192, 128),  # Adjusting for the correct number of input features (704, not 2240)
            nn.LayerNorm([128]),
            nn.ReLU(),
            nn.Linear(128, 128 * 4 * 4),  # This can remain as your design, assuming it matches your intended output
            nn.LayerNorm([128 * 4 * 4]),
            nn.ReLU(),
        )
        
        self.cnn_output = nn.Conv2d(128, 3, 1)
        self.opt = torch.optim.Adam(self.parameters(), lr=0.0008)


    def forward(self, x, x_ts):
        x_ts_processed = self.l_ts(x_ts)
        
        # Downsampling
        for block in self.down_blocks:
            x = block(x, x_ts_processed)
            x = F.max_pool2d(x, 2)

        x_flat = x.view(-1, 128 * 2 * 2)  # Flatten x for concatenation
        x_concat = torch.cat([x_flat, x_ts_processed], dim=1)  # Concatenate with time step information
        x_mlp = self.mlp(x_concat)

        # Reshape x after MLP for upsampling
        x = x_mlp.view(-1, 128, 4, 4)  # Ensure correct shape for upsampling

        # Upsampling
        for i, block in enumerate(self.up_blocks):
            x = block(x, x_ts_processed)
            if i < len(self.up_blocks) - 1:  # Avoid upsampling after the last block
                x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        x = self.cnn_output(x)
        return x
