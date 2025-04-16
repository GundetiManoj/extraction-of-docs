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
class AadhaarCard(BaseModel):
    name: str = Field(description="Full name of the Aadhaar cardholder")
    father_name: Optional[str] = Field(default=None, description="Father's name if available")
    dob: str = Field(description="Date of Birth in YYYY-MM-DD format")
    gender: str = Field(description="Gender (Male/Female/Other)")
    aadhaar_number: str = Field(description="12-digit Aadhaar number")
    address: Optional[str] = Field(default=None, description="Address if available")

# Function to Extract Aadhaar Card Data
def extract_aadhaar_data(file_path: str):
    if not os.path.exists(file_path):
        return {"error": "File does not exist"}

    file = client.files.upload(file=file_path, config={'display_name': file_path.split('/')[-1].split('.')[0]})

    prompt = "Extract all key details from the Aadhaar card and return structured JSON with keys: name, father_name, dob, gender, aadhaar_number, address."

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
file_path = "ITR DOC\BASIC\AADHAR CARD WITH DOB.jpeg"  # Raw string to avoid path issues

# Extract Data
extracted_aadhaar = extract_aadhaar_data(file_path)

# Save Extracted Data as JSON
output_json = "aadhaar_card_data_png.json"
with open(output_json, "w") as json_file:
    json.dump(extracted_aadhaar, json_file, indent=4)

print(f"Extraction Complete! JSON Saved: {output_json}")
