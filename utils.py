import torch
import os
import shutil
from torchvision import transforms
from torchvision.utils import save_image

def get_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),  # Resize images to a consistent size
        transforms.ToTensor(),  # Convert images to tensor
        transforms.Normalize((0.5,), (0.5,)),  # Normalize images
    ])

def save_generated_images(images, epoch, directory="generated_images", num_images=10):
    epoch_dir = os.path.join(directory, f'epoch_{epoch}')
    os.makedirs(epoch_dir, exist_ok=True)  # Create a new directory for the current epoch

    # Assuming images are in the range [-1, 1], denormalize to [0, 1]
    images = (images + 1) / 2.0

    # Ensure images tensor is correctly shaped as [B, C, H, W]
    # Reshape or view as necessary for your specific dimensions
    images = images.view(-1, 3, 256, 256)  # Adjust for your specific size and channels

    for i in range(num_images):
        save_path = os.path.join(epoch_dir, f"image_{i+1}.png")
        save_image(images[i], save_path)


def save_checkpoint(epoch, generator, discriminator, g_optimizer, d_optimizer, path, batch_idx=None):
    """Saves a checkpoint of the models and optimizers' states."""
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    temp_path = path + ".temp"
    torch.save({
        'epoch': epoch,
        'batch_idx': batch_idx,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'g_optimizer_state_dict': g_optimizer.state_dict(),
        'd_optimizer_state_dict': d_optimizer.state_dict(),
    }, temp_path)

    # Rename temp checkpoint to actual checkpoint path atomically
    shutil.move(temp_path, path)


def load_checkpoint(path, generator, discriminator, g_optimizer, d_optimizer):
    """Loads models and optimizers' states from a checkpoint."""
    checkpoint = torch.load(path)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
    d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
    epoch = checkpoint['epoch']
    batch_idx = checkpoint.get('batch_idx', -1)  # Default to -1 if not found
    return epoch, batch_idx
