import json
from google import genai
from pydantic import BaseModel, Field
from typing import List, Optional

from google import genai
# Create a client
api_key = "AIzaSyAxv3tpH8ZGdLMe6n8kseFDl2QxSGtan9M"
client = genai.Client(api_key=api_key)

# Define the model you are going to use
model_id =  "gemini-2.0-flash" # or "gemini-2.0-flash-lite-preview-02-05"  , "gemini-2.0-pro-exp-02-05"


# Configure Google Gemini API
genai.configure(api_key="AIzaSyAxv3tpH8ZGdLMe6n8kseFDl2QxSGtan9M")

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
    file = genai.upload_file(path=file_path, display_name="Aadhaar Card Image")
    prompt = "Extract all key details from the Aadhaar card and return structured JSON."

    response = genai.GenerativeModel("gemini-2.0-flash").generate_content(
        [prompt, file],
        generation_config={'response_mime_type': 'application/json'}
    )

    try:
        extracted_data = json.loads(response.text)
    except json.JSONDecodeError:
        extracted_data = {"error": "Failed to parse JSON response"}

    return extracted_data

# Aadhaar Image Path (Update with correct file path)
file_path = "Aadhar front_Khyati.jpeg"

# Extract Data
extracted_aadhaar = extract_aadhaar_data(file_path)

# Save Extracted Data as JSON
output_json = "aadhaar_card_data.json"
with open(output_json, "w") as json_file:
    json.dump(extracted_aadhaar, json_file, indent=4)

print(f"Extraction Complete! JSON Saved: {output_json}")
