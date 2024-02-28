# utils.py
import numpy as np
import torch
from tqdm.auto import trange
from torchvision.utils import make_grid
from PIL import Image
import os


timesteps = 16    # how many steps for a noisy image into clear
time_bar = 1 - np.linspace(0, 1.0, timesteps + 1) # linspace for timesteps


def save_images(images, epoch, directory="generated_diffusion_images", nmax=5):
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    images = images.detach().cpu()  # Ensure images are on CPU
    grid = make_grid(images[:nmax], nrow=8, normalize=True)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(os.path.join(directory, f"epoch_{epoch}.png"))


def forward_noise(x, t):
    a = time_bar[t]      # base on t
    b = time_bar[t + 1]  # image for t + 1
    
    noise = np.random.normal(size=x.shape)  # noise mask
    a = a.reshape((-1, 1, 1, 1))
    b = b.reshape((-1, 1, 1, 1))
    img_a = x * (1 - a) + noise * a
    img_b = x * (1 - b) + noise * b
    return img_a, img_b
    
def generate_ts(num):
    return np.random.randint(0, timesteps, size=num)

def train_one(x_img, model, device):
    x_ts = generate_ts(len(x_img))
    x_a, x_b = forward_noise(x_img, x_ts)
    
    x_ts = torch.from_numpy(x_ts).view(-1, 1).float().to(device)
    x_a = torch.tensor(x_a, dtype=torch.float).to(device)
    x_b = torch.tensor(x_b, dtype=torch.float).to(device)
    
    y_p = model(x_a, x_ts)
    loss = torch.mean(torch.abs(y_p - x_b))
    model.opt.zero_grad()
    loss.backward()
    model.opt.step()
    
    return loss.item()
