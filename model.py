import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, noise_vector_size, hidden_dim, image_size):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_vector_size, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, image_size),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, image_size, hidden_dim):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(image_size, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),  # Dropout can still be beneficial
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.net(x).squeeze()

