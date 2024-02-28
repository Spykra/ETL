import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, noise_vector_size, features_g=64, img_channels=3):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(noise_vector_size, features_g * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(features_g * 16),
            nn.SELU(True),
            # state size. (features_g*16) x 4 x 4
            nn.ConvTranspose2d(features_g * 16, features_g * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g * 8),
            nn.SELU(True),
            # state size. (features_g*8) x 8 x 8
            nn.ConvTranspose2d(features_g * 8, features_g * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g * 4),
            nn.SELU(True),
            # state size. (features_g*4) x 16 x 16
            nn.ConvTranspose2d(features_g * 4, features_g * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g * 2),
            nn.SELU(True),
            # state size. (features_g*2) x 32 x 32
            nn.ConvTranspose2d(features_g * 2, features_g, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g),
            nn.SELU(True),
            # state size. (features_g) x 64 x 64
            nn.ConvTranspose2d(features_g, features_g // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g // 2),
            nn.SELU(True),
            # state size. (features_g//2) x 128 x 128
            nn.ConvTranspose2d(features_g // 2, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (img_channels) x 256 x 256
        )

    def forward(self, input):
        return self.gen(input)
    

class Discriminator(nn.Module):
    def __init__(self, img_channels=3, features_d=64):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # input is (img_channels) x 256 x 256
            nn.Conv2d(img_channels, features_d, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (features_d) x 128 x 128
            nn.Conv2d(features_d, features_d * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_d * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (features_d*2) x 64 x 64
            nn.Conv2d(features_d * 2, features_d * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_d * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (features_d*4) x 32 x 32
            nn.Conv2d(features_d * 4, features_d * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_d * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (features_d*8) x 16 x 16
            nn.Conv2d(features_d * 8, features_d * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_d * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (features_d*16) x 8 x 8
            nn.Conv2d(features_d * 16, features_d * 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_d * 32),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (features_d*32) x 4 x 4
            nn.Conv2d(features_d * 32, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # state size. 1
        )

    def forward(self, input):
        return self.disc(input).view(-1)


