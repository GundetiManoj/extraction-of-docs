import json
import os
from google import genai
from pydantic import BaseModel, Field
from typing import List, Optional

# Initialize Gemini client
api_key = "AIzaSyAxv3tpH8ZGdLMe6n8kseFDl2QxSGtan9M"
client= genai.Client(api_key=api_key)
model_id="gemini-2.0-flash"
# Define Pydantic Schema for Aadhaar Card
class PANCardDetails(BaseModel):
    name: str = Field(description="Full Name of the PAN Cardholder")
    father_name: str = Field(description="Father's Name of the PAN Cardholder")
    date_of_birth: str = Field(description="Date of Birth in YYYY-MM-DD format")
    pan_number: str = Field(description="PAN Number")
    gender: Optional[str] = Field(default=None, description="Gender of the cardholder")

# Function to Extract Aadhaar Card Data
def extract_pan_data(file_path: str):
    if not os.path.exists(file_path):
        return {"error": "File does not exist"}

    file = client.files.upload(file=file_path, config={'display_name': file_path.split('/')[-1].split('.')[0]})

    prompt = "Extract all details from the PAN Card and return structured JSON."

    response = client.models.generate_content(
        model= model_id,
        contents=[prompt, file],
        config={'response_mime_type': 'application/json'}
    )

    try:
        extracted_data = json.loads(response.text)
    except json.JSONDecodeError:
        extracted_data = {"error": "Failed to parse JSON response"}

    return extracted_data

# Aadhaar Image Path (Update this to your actual file path)
file_path = "ITR DOC\BASIC\PAN.pdf"  # Raw string to avoid path issues

# Extract Data
extracted_pan = extract_pan_data(file_path)

# Save Extracted Data as JSON
output_json = "pan_card_data.json"
with open(output_json, "w") as json_file:
    json.dump(extracted_pan, json_file, indent=4)

print(f"Extraction Complete! JSON Saved: {output_json}")
