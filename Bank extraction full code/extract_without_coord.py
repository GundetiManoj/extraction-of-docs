import base64
import json
import os
import re
from typing import Dict, List, Any
from PIL import Image
from pdf2image import convert_from_path 
from google.generativeai import GenerativeModel, configure as configure_gemini
from google.cloud import documentai_v1 as documentai
from google.api_core.client_options import ClientOptions

# =================== CONFIGURATION ===================
GEMINI_API_KEY = "AIzaSyAxv3tpH8ZGdLMe6n8kseFDl2QxSGtan9M"
GOOGLE_APPLICATION_CREDENTIALS = "vision-454216-e49450484a5a.json"
PROJECT_ID = "753864940035"
LOCATION = "us"
PROCESSOR_ID = "f1ffe2360d31483c"
INPUT_DOCUMENT_PATH = r"ITR DOC\BANK STATEMENT\AXIS BANK STATEMENT.pdf"
OUTPUT_JSON_PATH = "document_extraction_results_pdf.json"
# =====================================================

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS

def setup_gemini_api():
    configure_gemini(api_key=GEMINI_API_KEY)

def encode_image_to_data_uri(image: Image.Image) -> str:
    import io
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_data = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_data}"

def convert_pdf_to_images(pdf_path: str) -> List[Image.Image]:
    return convert_from_path(pdf_path, dpi=200)

def extract_json_from_text(text: str) -> str:
    code_block_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if code_block_match:
        return code_block_match.group(1).strip()
    json_match = re.search(r"\{[\s\S]*\}", text)
    return json_match.group(0) if json_match else ""

def extract_personal_details(model: GenerativeModel, data_uri: str) -> Dict[str, str]:
    prompt = """
    Extract personal details (Name, DOB, Address, Phone, Email, ID numbers, etc.) from this document and return as JSON.
    """
    response = model.generate_content([prompt, {"mime_type": "image/png", "data": data_uri.split(",")[1]}])
    json_str = extract_json_from_text(response.text)
    return json.loads(json_str) if json_str else {}

def extract_tables(model: GenerativeModel, data_uri: str) -> List[Dict[str, List]]:
    prompt = """
    Extract all tables in this document. Return JSON with headers and rows for each table.
    Format: { "tables": [{ "headers": [...], "rows": [[...], [...]] }] }
    """
    response = model.generate_content([prompt, {"mime_type": "image/png", "data": data_uri.split(",")[1]}])
    json_str = extract_json_from_text(response.text)
    result = json.loads(json_str) if json_str else {}
    return result.get("tables", [])

def extract_key_value_pairs(model: GenerativeModel, data_uri: str) -> Dict[str, str]:
    prompt = """
    Extract all key-value pairs from this document. Return as JSON:
    { "keyValuePairs": { "key1": "value1", ... } }
    """
    response = model.generate_content([prompt, {"mime_type": "image/png", "data": data_uri.split(",")[1]}])
    json_str = extract_json_from_text(response.text)
    result = json.loads(json_str) if json_str else {}
    return result.get("keyValuePairs", {})

def extract_named_entities(file_path: str) -> List[Dict[str, Any]]:
    opts = ClientOptions(api_endpoint=f"{LOCATION}-documentai.googleapis.com")
    client = documentai.DocumentProcessorServiceClient(client_options=opts)
    name = f"projects/{PROJECT_ID}/locations/{LOCATION}/processors/{PROCESSOR_ID}"

    with open(file_path, "rb") as file:
        content = file.read()

    document = {"content": content, "mime_type": get_mime_type(file_path)}
    request = {"name": name, "raw_document": document}
    result = client.process_document(request=request)
    document = result.document

    entities = []
    for entity in document.entities:
        bounding_poly = []
        if entity.page_anchor and entity.page_anchor.page_refs:
            for page_ref in entity.page_anchor.page_refs:
                if page_ref.bounding_poly and page_ref.bounding_poly.normalized_vertices:
                    bounding_poly = [{"x": v.x, "y": v.y} for v in page_ref.bounding_poly.normalized_vertices]

        entities.append({
            "type": entity.type_,
            "value": entity.mention_text,
            "boundingPoly": bounding_poly or [{"x": 0, "y": 0}] * 4,
            "confidence": entity.confidence
        })

    return entities

def get_mime_type(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    return {
        '.pdf': 'application/pdf',
        '.png': 'image/png',
        '.jpg': 'image/jpg',
        '.jpeg': 'image/jpeg',
        '.tiff': 'image/tiff',
        '.bmp': 'image/bmp'
    }.get(ext, 'application/octet-stream')

def process_document() -> Dict[str, Any]:
    setup_gemini_api()
    model = GenerativeModel("gemini-2.0-flash")

    print(f"Processing document: {INPUT_DOCUMENT_PATH}")

    ext = os.path.splitext(INPUT_DOCUMENT_PATH)[1].lower()
    images = []

    if ext == '.pdf':
        try:
            images = convert_pdf_to_images(INPUT_DOCUMENT_PATH)
        except Exception as e:
            raise RuntimeError(f"Failed to convert PDF to images: {e}")
    else:
        try:
            img = Image.open(INPUT_DOCUMENT_PATH)
            images = [img]
        except Exception as e:
            raise RuntimeError(f"Failed to open image: {e}")

    combined_personal = {}
    combined_tables = []
    combined_kv_pairs = {}

    print("Extracting personal details, tables, key-value pairs using Gemini...")
    for idx, image in enumerate(images):
        print(f"Processing page {idx+1}...")
        data_uri = encode_image_to_data_uri(image)

        pd = extract_personal_details(model, data_uri)
        kv = extract_key_value_pairs(model, data_uri)
        tb = extract_tables(model, data_uri)

        if isinstance(pd, dict):
            combined_personal.update(pd)
        else:
            print(f"⚠️ Skipping personal details on page {idx+1}, invalid type: {type(pd)}")

        if isinstance(kv, dict):
            combined_kv_pairs.update(kv)
        else:
            print(f"⚠️ Skipping key-value pairs on page {idx+1}, invalid type: {type(kv)}")

        if isinstance(tb, list):
            combined_tables.extend(tb)
        else:
            print(f"⚠️ Skipping tables on page {idx+1}, invalid type: {type(tb)}")


    print("Extracting named entities using Document AI...")
    named_entities = extract_named_entities(INPUT_DOCUMENT_PATH)

    result = {
        "personalDetails": combined_personal,
        "tables": combined_tables,
        "keyValuePairs": combined_kv_pairs,
        "namedEntities": named_entities
    }

    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {OUTPUT_JSON_PATH}")
    return result

if __name__ == "__main__":
    try:
        output = process_document()
        print("Document processing complete.")
    except Exception as e:
        print(f"Error processing document: {e}")
