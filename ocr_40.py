import base64
import json
import os
from typing import Dict, List, Any
import re

# For Gemini API
from google.generativeai import GenerativeModel, configure as configure_gemini

# For Document AI
from google.cloud import documentai_v1 as documentai
from google.api_core.client_options import ClientOptions

# ============== CONFIGURATION - MODIFY THESE VALUES ==============
# Your Gemini API key
GEMINI_API_KEY = "AIzaSyAxv3tpH8ZGdLMe6n8kseFDl2QxSGtan9M"

# Path to your Google Cloud service account key file (for Document AI)
GOOGLE_APPLICATION_CREDENTIALS = "vision-454216-e49450484a5a.json"

# Document AI settings
PROJECT_ID = "753864940035"  # Your Google Cloud project ID
LOCATION = "us"  # Your Document AI processor location
PROCESSOR_ID = "f1ffe2360d31483c"  # Entity extraction processor

# Path to the input document
INPUT_DOCUMENT_PATH = "ITR DOC\BANK STATEMENT\download.png"

# Path where the output JSON file will be saved
OUTPUT_JSON_PATH = "document_extraction_results.json"
# ================================================================

# Set environment variable for Document AI authentication
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS

# Configure the Google Gemini API
def setup_gemini_api():
    configure_gemini(api_key=GEMINI_API_KEY)

# Function to encode document to base64
def encode_document_to_base64(file_path: str) -> str:
    with open(file_path, "rb") as file:
        file_data = file.read()
    
    # Determine MIME type based on file extension
    file_extension = os.path.splitext(file_path)[1].lower()
    mime_type = {
        '.pdf': 'application/pdf',
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.tiff': 'image/tiff',
        '.bmp': 'image/bmp'
    }.get(file_extension, 'application/octet-stream')
    
    # Create data URI
    base64_encoded = base64.b64encode(file_data).decode('utf-8')
    data_uri = f"data:{mime_type};base64,{base64_encoded}"
    
    return data_uri

# Extract personal details using Gemini
def extract_personal_details(model: GenerativeModel, document_data_uri: str) -> Dict[str, str]:
    prompt = """
    You are an expert in extracting personal information from documents.
    
    Analyze the document provided and extract all personal details such as:
    - Full name
    - Date of birth
    - Address
    - Phone number
    - Email
    - ID/Passport number
    - Any other personal identifiers
    
    Return the extracted information as a JSON object where each key is the field name and each value is the extracted information.
    
    Document: [Document will be provided]
    """
    
    response = model.generate_content(
        [prompt, {"mime_type": "image/png", "data": document_data_uri.split(",")[1]}]
    )
    
    response_text = response.text
    # Extract JSON from response
    json_str = extract_json_from_text(response_text)
    if json_str:
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return {"error": "Failed to parse personal details JSON"}
    return {}

# Extract tables using Gemini
def extract_tables(model: GenerativeModel, document_data_uri: str) -> List[Dict[str, List]]:
    prompt = """
    You are an expert in extracting tabular data from documents.
    
    Given the document provided, extract all tables present in the document. Return the tables in JSON format, with each table having a 'headers' array and a 'rows' array. Each row should be an array of strings.
    
    Return the tables in the following JSON format:
    {
      "tables": [
        {
          "headers": ["Header1", "Header2", ...],
          "rows": [
            ["Value1", "Value2", ...],
            ["Value1", "Value2", ...],
            ...
          ]
        },
        ...
      ]
    }
    
    Document: [Document will be provided]
    """
    
    response = model.generate_content(
        [prompt, {"mime_type": "image/png", "data": document_data_uri.split(",")[1]}]
    )
    
    response_text = response.text
    # Extract JSON from response
    json_str = extract_json_from_text(response_text)
    if json_str:
        try:
            result = json.loads(json_str)
            return result.get("tables", [])
        except json.JSONDecodeError:
            return []
    return []

# Extract key-value pairs using Gemini
def extract_key_value_pairs(model: GenerativeModel, document_data_uri: str) -> Dict[str, str]:
    prompt = """
    You are an expert form parser AI, your job is to extract key-value pairs from forms or documents.
    
    Analyze the document provided, and extract all key-value pairs that you can identify.
    Return the key-value pairs as a JSON object.
    
    The output should have the format:
    {
      "keyValuePairs": {
        "key1": "value1",
        "key2": "value2",
        ...
      }
    }
    
    Document: [Document will be provided]
    """
    
    response = model.generate_content(
        [prompt, {"mime_type": "image/png", "data": document_data_uri.split(",")[1]}]
    )
    
    response_text = response.text
    # Extract JSON from response
    json_str = extract_json_from_text(response_text)
    if json_str:
        try:
            result = json.loads(json_str)
            return result.get("keyValuePairs", {})
        except json.JSONDecodeError:
            return {}
    return {}

# Helper function to extract JSON from text responses
def extract_json_from_text(text: str) -> str:
    # Try to find JSON within markdown code blocks
    code_block_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if code_block_match:
        return code_block_match.group(1).strip()
    
    # Try to find JSON with curly braces
    json_match = re.search(r"\{[\s\S]*\}", text)
    if json_match:
        return json_match.group(0)
    
    return ""

# Extract named entities using Document AI
def extract_named_entities(file_path: str) -> List[Dict[str, Any]]:
    # Initialize Document AI client
    opts = ClientOptions(api_endpoint=f"{LOCATION}-documentai.googleapis.com")
    client = documentai.DocumentProcessorServiceClient(client_options=opts)
    
    # Full resource name of the processor
    name = f"projects/{PROJECT_ID}/locations/{LOCATION}/processors/{PROCESSOR_ID}"
    
    # Read the file into memory
    with open(file_path, "rb") as file:
        content = file.read()
    
    # Configure the process request
    document = {"content": content, "mime_type": get_mime_type(file_path)}
    request = {"name": name, "raw_document": document}
    
    # Process the document
    result = client.process_document(request=request)
    document = result.document
    
    # Extract entities with their bounding boxes
    entities = []
    for entity in document.entities:
        # Extract normalized vertices for the bounding box
        bounding_poly = []
        
        # Check if entity has a page_anchor with a bounding poly
        if entity.page_anchor and entity.page_anchor.page_refs:
            for page_ref in entity.page_anchor.page_refs:
                if page_ref.bounding_poly and page_ref.bounding_poly.normalized_vertices:
                    for vertex in page_ref.bounding_poly.normalized_vertices:
                        bounding_poly.append({"x": vertex.x, "y": vertex.y})
        
        # If no bounding poly found, provide empty list
        if not bounding_poly:
            bounding_poly = [{"x": 0, "y": 0}, {"x": 0, "y": 0}, {"x": 0, "y": 0}, {"x": 0, "y": 0}]
        
        entities.append({
            "type": entity.type_,
            "value": entity.mention_text,
            "boundingPoly": bounding_poly,
            "confidence": entity.confidence
        })
    
    return entities

# Helper function to get MIME type based on file extension
def get_mime_type(file_path: str) -> str:
    file_extension = os.path.splitext(file_path)[1].lower()
    mime_type = {
        '.pdf': 'application/pdf',
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.tiff': 'image/tiff',
        '.bmp': 'image/bmp'
    }.get(file_extension, 'application/octet-stream')
    return mime_type

# Main function to process document and extract all information
def process_document() -> Dict[str, Any]:
    # Setup Gemini API
    setup_gemini_api()
    
    # Load document and convert to data URI
    document_data_uri = encode_document_to_base64(INPUT_DOCUMENT_PATH)
    
    print(f"Processing document: {INPUT_DOCUMENT_PATH}")
    
    # Initialize Gemini model
    model = GenerativeModel('gemini-2.0-flash')
    
    # Extract information using Gemini and Document AI
    print("Extracting personal details...")
    personal_details = extract_personal_details(model, document_data_uri)
    
    print("Extracting tables...")
    tables = extract_tables(model, document_data_uri)
    
    print("Extracting key-value pairs...")
    key_value_pairs = extract_key_value_pairs(model, document_data_uri)
    
    print("Extracting named entities using Document AI...")
    named_entities = extract_named_entities(INPUT_DOCUMENT_PATH)
    
    # Combine all information into a single JSON
    result = {
        "personalDetails": personal_details,
        "tables": tables,
        "keyValuePairs": key_value_pairs,
        "namedEntities": named_entities
    }
    
    # Save to file
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as file:
        json.dump(result, file, indent=2, ensure_ascii=False)
    print(f"Results saved to {OUTPUT_JSON_PATH}")
    
    return result

if __name__ == "__main__":
    try:
        result = process_document()
        print("Document processing complete.")
    except Exception as e:
        print(f"Error processing document: {str(e)}")