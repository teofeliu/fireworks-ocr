import base64
import io
from fastapi import UploadFile
from schemas.result import Result
import torch
from torchvision import transforms
from PIL import Image
from app.models.rotation_model import load_rotation_model

class ImageProcessingService:
    def __init__(self):
        self.model = load_rotation_model('app/rotation_regression_model.pth')
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @staticmethod
    def encode_image(image_file: io.BytesIO) -> str:
        return base64.b64encode(image_file.read()).decode('utf-8')

    def preprocess_image(self, image_bytes):
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        long_side = max(img.size)
        square_img = Image.new('RGB', (long_side, long_side), (255, 255, 255))
        paste_coords = ((long_side - img.size[0]) // 2, (long_side - img.size[1]) // 2)
        square_img.paste(img, paste_coords)
        return self.transform(square_img)

    def predict_rotation(self, img_tensor):
        with torch.no_grad():
            output = self.model(img_tensor.unsqueeze(0)).squeeze()
            predicted_angle = output.item() * 345.0  # Denormalize
        return predicted_angle

    async def process_image(self, image: UploadFile, model_processor) -> Result:
        contents = await image.read()
        img_tensor = self.preprocess_image(contents)
        rotation_angle = self.predict_rotation(img_tensor)
        
        image_base64 = self.encode_image(io.BytesIO(contents))
        result = await model_processor(image_base64)
        
        # Add rotation angle to the result
        result.rotation_angle = rotation_angle
        
        return result