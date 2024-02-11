import torchvision.transforms as transforms
from load import load_images_from_folder

# Define a transform to convert the images to tensor and normalize
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Apply the transform to each image
training_glioma_tensors = [transform(image) for image in training_glioma_images]
