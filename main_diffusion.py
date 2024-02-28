# main.py
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from tqdm.auto import tqdm
import numpy as np

from diffusion_model import Model
from diffusion_utils import train_one, save_images

# Constants
IMG_SIZE = 64
BATCH_SIZE = 128

timesteps = 32    # how many steps for a noisy image into clear
time_bar = 1 - np.linspace(0, 1.0, timesteps + 1) # linspace for timesteps

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),  # Resize images to the expected input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Assuming 'Brain Tumor MRI Dataset/Training' is your folder path
dataset_path = "Brain Tumor MRI Dataset/Training"

train_dataset = ImageFolder(root=dataset_path, transform=transform)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# Model initialization
model = Model().to(device)

def train(model, trainloader, device, R=50, save_after_epochs=10):
    model.train()
    for epoch in range(R):
        total_loss = 0
        for batch_idx, (x_img, _) in enumerate(tqdm(trainloader, desc=f"Epoch {epoch+1}/{R}")):
            loss = train_one(x_img.to(device), model, device)
            total_loss += loss
        avg_loss = total_loss / len(trainloader)
        print(f"Epoch {epoch+1}: Average Loss: {avg_loss:.5f}")

        # Save images after specified epochs
        if (epoch + 1) % save_after_epochs == 0:
            with torch.no_grad():
                model.eval()  # Set the model to evaluation mode
                # Take a batch of images from the loader
                x_img, _ = next(iter(trainloader))
                x_img = x_img.to(device)
                # Generate images
                generated_images = model(x_img, torch.full([x_img.size(0), 1], timesteps-1, dtype=torch.float, device=device))
                # Save images
                save_images(generated_images, epoch+1)
                model.train()  # Set the model back to training mode



if __name__ == "__main__":
    train(model, trainloader, device, R=50, save_after_epochs=5)  # Corrected to save_after_epochs

    # Optional: Save your model
    torch.save(model.state_dict(), 'model_weights.pth')
