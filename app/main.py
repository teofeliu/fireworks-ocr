import logging
from fastapi import FastAPI
from routes import routes
from services.fireworks_client import get_fireworks_client
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

app = FastAPI()

# Include your routes
app.include_router(routes.router)

@app.on_event("startup")
async def startup_event():
    # Initialize Fireworks client
    get_fireworks_client()
    logging.info("Fireworks client initialized")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)