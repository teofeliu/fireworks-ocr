import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import matplotlib.pyplot as plt

# Define your model architecture
class RotationRegressionModel(nn.Module):
    def __init__(self):
        super(RotationRegressionModel, self).__init__()
        self.resnet = models.resnet18(pretrained=False)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.resnet(x)

def load_and_preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    long_side = max(img.size)
    square_img = Image.new('RGB', (long_side, long_side), (255, 255, 255))
    paste_coords = ((long_side - img.size[0]) // 2, (long_side - img.size[1]) // 2)
    square_img.paste(img, paste_coords)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return transform(square_img), square_img

def predict_rotations(model, image_folder):
    model.eval()
    results = []
    
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder, filename)
            img_tensor, original_img = load_and_preprocess_image(image_path)
            
            with torch.no_grad():
                output = model(img_tensor.unsqueeze(0)).squeeze()
                predicted_angle = output.item() * 345.0  # Denormalize
            
            results.append((filename, original_img, predicted_angle))
    
    return results

def display_results(results):
    num_images = len(results)
    fig, axes = plt.subplots(num_images, 1, figsize=(10, 5 * num_images))
    if num_images == 1:
        axes = [axes]
    
    for ax, (filename, img, angle) in zip(axes, results):
        ax.imshow(img)
        ax.set_title(f"{filename}\nPredicted Rotation: {angle:.2f} degrees")
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the model
    model = RotationRegressionModel()
    model.load_state_dict(torch.load('rotation_regression_model.pth', map_location=device))
    model = model.to(device)

    # Specify the folder containing your test images
    image_folder = 'path/to/your/test/images'

    # Predict rotations for all images in the folder
    results = predict_rotations(model, image_folder)

    # Display the results
    display_results(results)