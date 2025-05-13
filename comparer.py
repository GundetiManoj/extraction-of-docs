import json
import re
import google.generativeai as genai
from difflib import SequenceMatcher

# === Initialize Gemini Flash with API Key ===
genai.configure(api_key="AIzaSyAxv3tpH8ZGdLMe6n8kseFDl2QxSGtan9M")
model = genai.GenerativeModel("gemini-1.5-flash")

# === Utility Functions ===

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def normalize(text):
    return re.sub(r"[^A-Za-z0-9]+", "", text).lower()

def match_text_to_coordinates(entity_text, ocr_text_blocks):
    best_match = None
    highest_ratio = 0
    entity_norm = normalize(entity_text)

    for block in ocr_text_blocks:
        ocr_text = block.get("text", "")
        ratio = SequenceMatcher(None, entity_norm, normalize(ocr_text)).ratio()
        if ratio > highest_ratio:
            highest_ratio = ratio
            best_match = {
                "text": ocr_text,
                "bounding_box": block.get("bounding_box", []),
                "page": block.get("page", 1),
                "confidence": round(ratio, 2)
            }

    return best_match if highest_ratio > 0.5 else None

def llm_refine_coordinates(entity_label, entity_value, best_match):
    prompt = f"""
You are a document understanding AI.

A field labeled "{entity_label}" has the expected value "{entity_value}".
The best OCR match is: "{best_match.get('text')}" from page {best_match.get('page')} with bounding box: {best_match.get('bounding_box')}.

Return a JSON object like:
{{
  "field": "...",
  "value": "...",
  "bounding_box": [...],
  "page": ...,
  "confidence": ...
}}

Make small corrections to bounding box and confidence if needed.
"""
    try:
        response = model.generate_content(prompt)
        return json.loads(response.text)
    except Exception as e:
        return {
            "field": entity_label,
            "value": entity_value,
            "bounding_box": best_match.get("bounding_box", []),
            "page": best_match.get("page", 1),
            "confidence": best_match.get("confidence", 0.5)
        }

# === Core Processing ===

def compare_and_structure(reference_data, extracted_data):
    ocr_blocks = extracted_data.get("text_coordinates", [])

    result = {
        "personal_details": [],
        "key_value_pairs": [],
        "named_entities": [],
        "tables": []
    }

    # Process personalDetails
    for key, value in reference_data.get("personalDetails", {}).items():
        match = match_text_to_coordinates(value, ocr_blocks)
        if match:
            structured = llm_refine_coordinates(key, value, match)
            result["personal_details"].append(structured)
            result["key_value_pairs"].append(structured)

    # Tables
    for table in reference_data.get("tables", []):
        table_rows = []
        headers = table.get("headers", [])
        for row in table.get("rows", []):
            structured_row = []
            for cell in row:
                if cell:
                    match = match_text_to_coordinates(cell, ocr_blocks)
                    structured_cell = llm_refine_coordinates("table_cell", cell, match) if match else {
                        "value": cell, "bounding_box": [], "page": None, "confidence": 0.0
                    }
                    structured_row.append(structured_cell)
                else:
                    structured_row.append({"value": "", "bounding_box": [], "page": None, "confidence": 0.0})
            table_rows.append(structured_row)
        result["tables"].append({"headers": headers, "rows": table_rows})

    # Named Entity Recognition (NER): reuse personal details
    for entity in result["personal_details"]:
        result["named_entities"].append({
            "entity_type": entity["field"],
            "value": entity["value"],
            "bounding_box": entity["bounding_box"],
            "page": entity["page"],
            "confidence": entity["confidence"]
        })

    return result

# === Entry Point ===

if __name__ == "__main__":
    reference_path = r"document_extraction_results.json"  # Path to the reference JSON file without coordinates
    
    extracted_path = r"exp_statement_output.json" # Path to the extracted JSON file with coordinates
    output_path = r"final_comparison.json"

    ref_data = load_json(reference_path)
    ext_data = load_json(extracted_path)

    final_result = compare_and_structure(ref_data, ext_data)
    save_json(final_result, output_path)

    print(f"✅ Final comparison saved to {output_path}")
