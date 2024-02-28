import os
import torch
import time
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader
from model_full_connected import Generator, Discriminator
from utils import get_transform, save_generated_images, save_checkpoint, load_checkpoint

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
lr = 0.0001
batch_size = 32
epochs = 100
hidden_dim = 256
noise_vector_size = 100
image_size = 256 * 256 * 3  # For 256x256 RGB images
lambda_gp = 10  # Gradient penalty lambda hyperparameter
n_critic = 5  # The number of discriminator updates per generator update

# Initialize models
generator = Generator(noise_vector_size, hidden_dim, image_size).to(device)
discriminator = Discriminator(image_size, hidden_dim).to(device)

# Adjusting learning rates
g_lr = 0.0002
d_lr = 0.0002

# Optimizers
g_optimizer = optim.Adam(generator.parameters(), lr=g_lr, betas=(0.5, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=d_lr, betas=(0.5, 0.999))

# Learning rate schedulers
g_scheduler = torch.optim.lr_scheduler.StepLR(g_optimizer, step_size=10, gamma=0.5)
d_scheduler = torch.optim.lr_scheduler.StepLR(d_optimizer, step_size=10, gamma=0.5)

# Data loading
dataset = datasets.ImageFolder(root='Brain Tumor MRI Dataset/Training', transform=get_transform())
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Calculate Gradient Penalty
def compute_gradient_penalty(D, real_samples, fake_samples):
    batch_size = real_samples.size(0)
    alpha = torch.rand(batch_size, 1, device=real_samples.device).expand_as(real_samples)
    interpolates = alpha * real_samples + (1 - alpha) * fake_samples
    interpolates = interpolates.requires_grad_(True)
    d_interpolates = D(interpolates)
    
    grad_outputs = torch.ones_like(d_interpolates)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    # Reshape gradients to calculate norm over all dimensions except the batch dimension
    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
    return gradient_penalty

checkpoint_dir = './training_checkpoints'
checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_epoch_75.pth')

start_epoch = 0
if os.path.isfile(checkpoint_path):
    start_epoch = load_checkpoint(checkpoint_path, generator, discriminator, g_optimizer, d_optimizer)

for epoch in range(start_epoch, epochs):
    start_time = time.time()
    
    for i, (imgs, _) in enumerate(dataloader):
        imgs = imgs.to(device)
        real_imgs = imgs.view(imgs.size(0), -1)
        current_batch_size = real_imgs.size(0)
        
        # Label smoothing for discriminator
        real_labels = 0.9 * torch.ones(current_batch_size, 1).to(device)  # Soft labels for real images
        fake_labels = torch.zeros(current_batch_size, 1).to(device)  # Hard labels for fake images
        
        # Train Discriminator
        d_optimizer.zero_grad()
        z = torch.randn(current_batch_size, noise_vector_size, device=device)
        fake_imgs = generator(z)

        # Ensure the discriminator's outputs are squeezed to match the label's shape
        real_loss = torch.nn.functional.binary_cross_entropy_with_logits(discriminator(real_imgs).squeeze(), real_labels.squeeze())
        fake_loss = torch.nn.functional.binary_cross_entropy_with_logits(discriminator(fake_imgs.detach()).squeeze(), fake_labels.squeeze())
        d_loss = real_loss + fake_loss

        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1)  # Gradient clipping for discriminator
        d_optimizer.step()
        
        # Train Generator every n_critic steps
        if i % n_critic == 0:
            g_optimizer.zero_grad()

            # Fresh fake images for generator update
            fake_imgs = generator(z)

            # Soft labels for fake images to fool the discriminator
            gen_labels = 0.9 * torch.ones(current_batch_size, 1).to(device)  # Soft labels to encourage the generator

            # Calculating the generator's loss
            # Ensure the discriminator's outputs and labels have matching dimensions
            gen_loss = torch.nn.functional.binary_cross_entropy_with_logits(discriminator(fake_imgs).squeeze(), gen_labels.squeeze())

            gen_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1)  # Gradient clipping for generator
            g_optimizer.step()

        if i % 40 == 0:
            print(f"[Epoch {epoch+1}/{epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item()}] [G loss: {gen_loss.item()}] [D(x): {real_loss.item()}] [D(G(z)): {fake_loss.item()}]")

    g_scheduler.step()
    d_scheduler.step()

    if epoch % 5 == 0:
        save_checkpoint(epoch, generator, discriminator, g_optimizer, d_optimizer, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth'))
        
    print(f"Epoch {epoch+1}/{epochs} completed in {time.time() - start_time:.2f} seconds.")
    save_generated_images(fake_imgs.detach(), epoch, directory="generated_images", num_images=10)
