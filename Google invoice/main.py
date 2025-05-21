from google.cloud import documentai_v1 as documentai
import os
import json
import google.generativeai as genai

# Set credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "vision-454216-e49450484a5a.json"

PROJECT_ID = "vision-454216"
LOCATION = "us"
PROCESSOR_ID = "6f74c03880d3a595"
GEMINI_API_KEY = "AIzaSyAxv3tpH8ZGdLMe6n8kseFDl2QxSGtan9M"

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
                "page_number": page_index + 1
            })
    return result

def extract_key_value_pairs(document):
    kvs = []
    for entity in document.entities:
        if not entity.page_anchor.page_refs:
            continue
        for ref in entity.page_anchor.page_refs:
            if ref.bounding_poly:
                bbox = get_normalized_bbox(ref.bounding_poly)
                kvs.append({
                    "key": entity.type_,
                    "value": entity.mention_text,
                    "confidence": entity.confidence,
                    "bounding_box": bbox,
                    "page_number": ref.page + 1
                })
    return kvs

def extract_tables(document):
    tables = []
    for page_index, page in enumerate(document.pages):
        for table in page.tables:
            structured_table = []
            for row in table.header_rows + table.body_rows:
                row_data = []
                for cell in row.cells:
                    row_data.append({
                        "text": cell.layout.text_anchor.content.strip(),
                        "bounding_box": get_normalized_bbox(cell.layout.bounding_poly),
                        "confidence": cell.layout.confidence
                    })
                structured_table.append(row_data)
            tables.append(structured_table)
    return tables

def extract_ner(document):
    ner_data = []
    for entity in document.entities:
        for ref in entity.page_anchor.page_refs:
            if ref.bounding_poly:
                ner_data.append({
                    "text": entity.mention_text,
                    "type": entity.type_,
                    "confidence": entity.confidence,
                    "bounding_box": get_normalized_bbox(ref.bounding_poly),
                    "page_number": ref.page + 1
                })
    return ner_data

def call_gemini_for_extraction(text):
    prompt = f"""
Extract the following from this invoice:
- Personal details (Name, Address, Phone, Email, GSTIN, etc.)
- Named entities (person, organization, location, dates, money, etc.)
- All tables (structured in JSON)

Output JSON format:
{{
  "personal_details": {{ }},
  "named_entities": [ ],
  "tables": [ ]
}}

Text Content:
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
        tables = gemini_data.get("tables", [])

    final_output = {
        "text_with_coords": text_coords,
        "key_value_pairs": kv_pairs,
        "personal_details": gemini_data.get("personal_details", {}),
        "named_entities": gemini_data.get("named_entities", []),
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
    input_path = r"ITR DOC\EMPLOYEE INFO\FORM 16.pdf"
    output_path = "output_result.json"
    result = analyze_invoice(input_path)
    save_output(result, output_path)
    print(f"✅ Output saved to: {output_path}")