# app/services/image_processing.py
import base64
import io
from fastapi import UploadFile
from schemas.result import Result

# Image orientation correction imports and functions
USE_ORIENTATION_CORRECTION = False  # Set to False to disable orientation correction

if USE_ORIENTATION_CORRECTION:
    import cv2
    import pytesseract
    import numpy as np

    def rotate_image(image, angle):
        return cv2.rotate(image, angle)

    def get_text_density(image):
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Get bounding boxes of text
        boxes = pytesseract.image_to_boxes(gray)
        # Calculate the density of text (number of bounding boxes)
        density = len(boxes.splitlines())
        return density

    def correct_orientation(image_bytes):
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        # Decode numpy array to OpenCV image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Possible rotations
        rotations = [
            (cv2.ROTATE_90_CLOCKWISE, 90),
            (cv2.ROTATE_180, 180),
            (cv2.ROTATE_90_COUNTERCLOCKWISE, 270),
            (None, 0)  # No rotation
        ]
        
        best_density = -1
        best_image = image
        best_orientation = 0
        
        # Check each rotation
        for rotate, angle in rotations:
            if rotate is not None:
                rotated_image = rotate_image(image, rotate)
            else:
                rotated_image = image
            density = get_text_density(rotated_image)
            if density > best_density:
                best_density = density
                best_image = rotated_image
                best_orientation = angle
        
        print(f"Best orientation: {best_orientation}°")
        
        # Encode the image back to bytes
        _, img_encoded = cv2.imencode('.png', best_image)
        return img_encoded.tobytes()

class ImageProcessingService:
    @staticmethod
    def encode_image(image_file: io.BytesIO) -> str:
        return base64.b64encode(image_file.read()).decode('utf-8')

    @staticmethod
    async def process_image(image: UploadFile, model_processor) -> Result:
        contents = await image.read()
        
        if USE_ORIENTATION_CORRECTION:
            contents = correct_orientation(contents)
        
        image_base64 = ImageProcessingService.encode_image(io.BytesIO(contents))
        return await model_processor(image_base64)