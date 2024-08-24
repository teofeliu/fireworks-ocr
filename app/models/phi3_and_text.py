from schemas.result import Result
from services.fireworks_client import get_fireworks_client, MODEL_PHI_3
import json
import logging

async def process(image_base64: str):
    client = get_fireworks_client()
    try:
        response = client.completion.create(
            model = "accounts/fireworks/models/phi-3-vision-128k-instruct",
            prompt = "Q: <|image|>\n Extract ALL of the words and numbers in the document. Only include words, numbers, and dates. Do this in one line, separated by commas \n A: Text: ",
            images = [f"data:image/png;base64,{image_base64}"],
            max_tokens = 5000,
            temperature = 0.1
        )

        response = client.completion.create(
            model="accounts/fireworks/models/llama-v3-8b-instruct",
            response_format={"type": "json_object", "schema": Result.model_json_schema()},
            prompt = f"Q: Here is text parsed from an identification document. Here is the text: {response.choices[0].text} \n A: Text: ",
            max_tokens = 1000
        )

                # Log the raw response
        logging.info(f"Raw response: {response.choices[0].text}")
        
        # Parse the JSON response
        try:
            result_dict = json.loads(response.choices[0].text)
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON: {e}")
            logging.error(f"Raw text: {response.choices[0].text}")
            raise ValueError("Invalid JSON response from model")

        # Log the parsed dictionary
        logging.info(f"Parsed result_dict: {result_dict}")
        
        # Ensure all required fields are present
        required_fields = ['first_name', 'last_name', 'date_of_birth', 'id_number']
        for field in required_fields:
            if field not in result_dict:
                result_dict[field] = "N/A"  # or some default value
        
        # Create and return a Result object
        return Result(**result_dict)
    except Exception as e:
        logging.error(f"Error in process function: {str(e)}")
        raise