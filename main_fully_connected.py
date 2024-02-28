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
epochs = 50
hidden_dim = 256
noise_vector_size = 100
image_size = 256 * 256 * 3  # For 256x256 RGB images
lambda_gp = 10  # Gradient penalty lambda hyperparameter
n_critic = 5  # The number of discriminator updates per generator update

# Initialize models and move them to the correct device
generator = Generator(noise_vector_size, hidden_dim, image_size).to(device)
discriminator = Discriminator(image_size, hidden_dim).to(device)

# Adjusting learning rates
g_lr = 0.0002  # Adjusted learning rate for generator
d_lr = 0.0002  # Adjusted learning rate for discriminator

g_optimizer = optim.Adam(generator.parameters(), lr=g_lr, betas=(0.5, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=d_lr, betas=(0.5, 0.999))

# Adding a learning rate scheduler
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
checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_epoch_10.pth')  # Corrected to check for specific checkpoint file

start_epoch = 0
start_batch_idx = 0  # Starting from the first batch

# Correctly check for the specific checkpoint file
if os.path.isfile(checkpoint_path):
    # start_epoch, start_batch_idx = load_checkpoint(checkpoint_path, generator, discriminator, g_optimizer, d_optimizer)
    start_epoch = load_checkpoint(checkpoint_path, generator, discriminator, g_optimizer, d_optimizer)

for epoch in range(start_epoch, epochs):
    start_time = time.time()  # To track time per epoch
    
    for i, (imgs, _) in enumerate(dataloader):
        if epoch == start_epoch and i < start_batch_idx:
            continue 

        imgs = imgs.to(device)
        real_imgs = imgs.view(imgs.size(0), -1)

        current_batch_size = real_imgs.size(0)  # Dynamic batch size

        # Train Discriminator
        d_optimizer.zero_grad()
        z = torch.randn(current_batch_size, noise_vector_size, device=device)  # Use current_batch_size
        fake_imgs = generator(z).detach()
        real_validity = discriminator(real_imgs)
        fake_validity = discriminator(fake_imgs)
        gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data)
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + gradient_penalty
        d_loss.backward()
        d_optimizer.step()

        # Calculate D(x) and D(G(z)) for printing
        with torch.no_grad():  # We don't need gradients for this part
            # For real images
            D_x = real_validity.mean().item()
            # For fake images
            D_G_z = fake_validity.mean().item()

        # Train Generator less frequently
        if i % n_critic == 0:
            g_optimizer.zero_grad()
            # We need fresh fake images here
            gen_imgs = generator(z)
            gen_loss = -torch.mean(discriminator(gen_imgs))
            gen_loss.backward()
            g_optimizer.step()

        if i % 40 == 0:
            print(f"[Epoch {epoch+1}/{epochs}] [Batch {i}/{len(dataloader)}] "
                f"[D loss: {d_loss.item()}] [G loss: {gen_loss.item()}] "
                f"[D(x): {D_x}] [D(G(z)): {D_G_z}]")

    if epoch % 5 == 0:
        historical_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        save_checkpoint(epoch, generator, discriminator, g_optimizer, d_optimizer, historical_path)

    g_scheduler.step()
    d_scheduler.step()
    start_batch_idx = 0  # Reset for the next epoch

    elapsed_time = time.time() - start_time  # Calculate elapsed time
    print(f"Epoch {epoch+1}/{epochs} completed in {elapsed_time:.2f} seconds.")

    # Save generated images at the end of each epoch
    save_generated_images(gen_imgs.detach(), epoch, directory="generated_images", num_images=10)


# d_loss: The loss of the discriminator should ideally be low but not zero. A low loss indicates that the discriminator 
# is confidently distinguishing between real and fake images. A loss of zero, as seen in epochs 4 and 5, suggests that 
# the discriminator is too confident, which might indicate overfitting or a failure mode.

# g_loss: The loss of the generator should ideally decrease over time. This loss measures how well the generator 
# is fooling the discriminator. However, extremely high values, as seen in your output, suggest that the generator 
# is not performing well.

# D(x): This value should ideally be close to 1, indicating that the discriminator correctly identifies real images as real.
#  Consistently high values close to 1, as seen in your results, indicate that the discriminator is performing its task 
# correctly for real images.

# D(G(z)): This value indicates the discriminator's output for fake images. Early in training, you would expect 
#    this to be closer to 0, as the discriminator can easily tell that the generator's outputs are fake. 
# Over time, as the generator improves, you would expect this value to rise towards 1, indicating that the discriminator 
# is being fooled. A value that remains at 0 indicates the generator is not improving and the discriminator can easily 
# distinguish all fake images.

# Based on the progression you've shown:

# The discriminator is becoming too confident (loss approaching 0), which is a sign that it's potentially overpowering 
# the generator and not providing useful gradients for the generator to learn from.
# The generator loss is very high and increasing, suggesting that it's not successfully fooling the discriminator.