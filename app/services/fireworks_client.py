# app/services/fireworks_client.py

from fireworks.client import Fireworks
import os

# Initialize the Fireworks client
fireworks_client = Fireworks(api_key=Fireworks(api_key=os.getenv("FIREWORKS_API_KEY")))

# Model constants
MODEL_PHI_3 = "accounts/fireworks/models/phi-3-vision-128k-instruct"

def get_fireworks_client():
    return fireworks_client