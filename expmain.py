from google.cloud import documentai_v1 as documentai
import os
import json
import google.generativeai as genai
# Environment setup
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "vision-454216-e49450484a5a.json"
PROJECT_ID = "vision-454216"
LOCATION = "us"
PROCESSOR_ID = "e8d7e54d9f2335a3"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
def process_document(file_path: str, mime_type: str = "application/pdf"):
    client = documentai.DocumentProcessorServiceClient()
    name = f"projects/{PROJECT_ID}/locations/{LOCATION}/processors/{PROCESSOR_ID}"
    with open(file_path, "rb") as file:
        input_doc = file.read()
    raw_document = documentai.RawDocument(content=input_doc, mime_type=mime_type)
    request = documentai.ProcessRequest(name=name, raw_document=raw_document)
    result = client.process_document(request=request)
    return result.document
def get_normalized_bbox(poly):
    if not poly.normalized_vertices:
        return {}
    verts = poly.normalized_vertices
    return {
        "x1": round(verts[0].x, 4), "y1": round(verts[0].y, 4),
        "x2": round(verts[1].x, 4), "y2": round(verts[1].y, 4),
        "x3": round(verts[2].x, 4), "y3": round(verts[2].y, 4),
        "x4": round(verts[3].x, 4), "y4": round(verts[3].y, 4)
    }
def get_text_from_anchor(text_anchor, full_text):
    segments = text_anchor.text_segments
    return "".join([full_text[seg.start_index:seg.end_index] for seg in segments]).strip()

def get_layout_info(text_anchor, doc):
    """Get page number and bounding box from text anchor."""
    if not text_anchor.text_segments:
        return {"page_number": None, "bounding_box": {}}    
    for page in doc.pages:
        for token in page.tokens:
            if token.layout.text_anchor.text_segments:
                token_seg = token.layout.text_anchor.text_segments[0]
                anchor_seg = text_anchor.text_segments[0]
                if token_seg.start_index == anchor_seg.start_index:
                    return {
                        "page_number": page.page_number,
                        "bounding_box": get_normalized_bbox(token.layout.bounding_poly)
                    }
    return {"page_number": None, "bounding_box": {}}

def extract_key_value_pairs(doc):
    kv_pairs = []
    full_text = doc.text
    for entity in doc.entities:
        key = entity.type_
        value_text = get_text_from_anchor(entity.text_anchor, full_text)
        info = get_layout_info(entity.text_anchor, doc)
        kv_pairs.append({
            "field": key,
            "value": value_text,
            "page_number": info["page_number"],
            "bounding_box": info["bounding_box"]
        })
        # Nested properties (e.g., line item fields)
        for prop in entity.properties:
            sub_key = f"{key}.{prop.type_}"
            sub_val = get_text_from_anchor(prop.text_anchor, full_text)
            sub_info = get_layout_info(prop.text_anchor, doc)
            kv_pairs.append({
                "field": sub_key,
                "value": sub_val,
                "page_number": sub_info["page_number"],
                "bounding_box": sub_info["bounding_box"]
            })

    return kv_pairs

def extract_named_entities(doc):
    named_entities = []
    full_text = doc.text
    for entity in doc.entities:
        if not entity.text_anchor.text_segments:
            continue  # Skip if no text segments
        text = get_text_from_anchor(entity.text_anchor, full_text)
        layout_info = get_layout_info(entity.text_anchor, doc)
        named_entities.append({
            "type": entity.type_,
            "text": text,
            "confidence": entity.confidence,
            "page_number": layout_info.get("page_number", -1),
            "bounding_box": layout_info.get("bounding_box", [])
        })
    return named_entities

def extract_text_with_coords(document):
    result = []
    for page_index, page in enumerate(document.pages):
        full_text = document.text
        for token in page.tokens:
            if not token.layout.text_anchor.text_segments:
                continue
            segment = token.layout.text_anchor.text_segments[0]
            text = full_text[segment.start_index:segment.end_index]
            result.append({
                "text": text.strip(),
                "bounding_box": get_normalized_bbox(token.layout.bounding_poly),
                "confidence": token.layout.confidence,
                "page_number": page.page_number
            })
    return result

def get_text(layout, full_text):
    if not layout.text_anchor.text_segments:
        return ""
    return "".join([full_text[seg.start_index:seg.end_index] for seg in layout.text_anchor.text_segments]).strip()

def extract_tables(doc):
    tables = []
    for page in doc.pages:
        for table in page.tables:
            for row in list(table.header_rows) + list(table.body_rows):
                row_data = []
                for cell in row.cells:
                    row_data.append({
                        "text": get_text(cell.layout, doc.text),
                        "page_number": page.page_number,
                        "bounding_box": get_normalized_bbox(cell.layout.bounding_poly)
                    })
                tables.append(row_data)
    return tables

def call_gemini_for_extraction(text):
    prompt = f"""
Extract:
- Personal details (name, phone, email, GSTIN, etc.)
- Named entities (person, org, date, money, location)
Format:
{{
  "personal_details": {{ }},
  "named_entities": [ ]
}}
Text:
{text}
    """
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)

    try:
        cleaned = response.text.strip()
        if cleaned.startswith("```") and cleaned.endswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[1:-1])
        return json.loads(cleaned)
    except Exception as e:
        print("[Gemini ERROR]", e)
        print("Raw Gemini Response:", response.text)
        return {"personal_details": {}, "named_entities": [], "tables": []}

def analyze_invoice(file_path, mime_type="application/pdf"):
    doc = process_document(file_path, mime_type)
    text_coords = extract_text_with_coords(doc)
    all_text = " ".join([t["text"] for t in text_coords])
    
    gemini_data = call_gemini_for_extraction(all_text)
    kv_pairs = extract_key_value_pairs(doc)
    if not kv_pairs:
        print("[INFO] No KV pairs from Document AI")

    tables = extract_tables(doc)
    if not tables:
        print("[INFO] No tables found by Document AI, using Gemini output.")
        tables = gemini_data.get("tables", [])

    final_output = {
        "text_with_coords": text_coords,
        "key_value_pairs": kv_pairs,
        "personal_details": gemini_data.get("personal_details", {}),
        "named_entities": extract_named_entities(doc),
        "tables": tables
    }

    with open("invoice_output.json", "w") as f:
        json.dump(final_output, f, indent=2)

    return final_output

def save_output(result, out_path):
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

# MAIN
if __name__ == "__main__":
    input_path = r"ITR DOC\BANK STATEMENT\AXIS BANK STATEMENT.pdf"
    output_path = "axisbank_with_coords.json"
    result = analyze_invoice(input_path)
    save_output(result, output_path)
    print(f"✅ Output saved to: {output_path}")
