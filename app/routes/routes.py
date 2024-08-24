from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from services.image_processing import ImageProcessingService
from models import phi3, phi3_and_text
import logging

router = APIRouter()

@router.post("/process_image/phi3")
async def process_image_phi3(image: UploadFile = File(...)):
    """
    Process an uploaded image using the Phi-3 model.
    - **image**: The image file to be processed
    Returns the extracted information as a Result object.
    """
    try:
        result = await ImageProcessingService.process_image(image, phi3.process)
        if isinstance(result, dict):
            return JSONResponse(content=result, status_code=200)
        else:
            return JSONResponse(content=result.dict(), status_code=200)
    except Exception as e:
        logging.error(f"Error processing image: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@router.post("/process_image/phi3_and_text")
async def process_image_phi3_and_text(image: UploadFile = File(...)):
    """
    Process an uploaded image using the Phi-3 model.
    - **image**: The image file to be processed
    Returns the extracted information as a Result object.
    """
    try:
        result = await ImageProcessingService.process_image(image, phi3_and_text.process)
        if isinstance(result, dict):
            return JSONResponse(content=result, status_code=200)
        else:
            return JSONResponse(content=result.dict(), status_code=200)
    except Exception as e:
        logging.error(f"Error processing image: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)