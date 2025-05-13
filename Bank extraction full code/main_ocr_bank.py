from typing import List, Dict, Any
from google.api_core.client_options import ClientOptions
from google.cloud import documentai_v1 as documentai
import os
import json

def process_document(project_id: str, location: str, processor_id: str, file_path: str, mime_type: str) -> documentai.Document:
    client = documentai.DocumentProcessorServiceClient(
        client_options=ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")
    )
    name = client.processor_path(project_id, location, processor_id)
    with open(file_path, "rb") as f:
        raw_document = documentai.RawDocument(content=f.read(), mime_type=mime_type)
    request = documentai.ProcessRequest(name=name, raw_document=raw_document)
    result = client.process_document(request=request)
    return result.document

def get_text(doc: documentai.Document, element: documentai.Document.TextAnchor) -> str:
    return "".join(
        doc.text[seg.start_index:seg.end_index]
        for seg in element.text_segments
    ).strip()

def extract_text_with_coordinates(document: documentai.Document) -> List[Dict[str, Any]]:
    output = []
    for page_index, page in enumerate(document.pages, start=1):
        for line in page.lines:
            text = get_text(document, line.layout.text_anchor)
            bbox = [(v.x, v.y) for v in line.layout.bounding_poly.vertices]
            output.append({
                "text": text,
                "bounding_box": [{"x": x, "y": y} for x, y in bbox],
                "page": page_index
            })
    return output

def extract_key_value_pairs(document: documentai.Document) -> List[Dict[str, Any]]:
    kv_pairs = []
    for page_index, page in enumerate(document.pages, start=1):
        for field in page.form_fields:
            key = get_text(document, field.field_name.text_anchor)
            value = get_text(document, field.field_value.text_anchor)
            bbox = [(v.x, v.y) for v in field.field_name.bounding_poly.vertices]
            confidence = field.field_name.confidence
            kv_pairs.append({
                "type": key.lower().replace(" ", "_"),
                "value": value,
                "confidence": round(confidence, 2),
                "page": page_index,
                "bounding_box": [{"x": v[0], "y": v[1]} for v in bbox]
            })
    return kv_pairs

def extract_transactions(document: documentai.Document) -> List[Dict[str, str]]:
    transactions = []
    for page in document.pages:
        for table in page.tables:
            headers = []
            if table.header_rows:
                headers = [get_text(document, cell.layout.text_anchor) for cell in table.header_rows[0].cells]
            for row in table.body_rows:
                row_data = {}
                for idx, cell in enumerate(row.cells):
                    text = get_text(document, cell.layout.text_anchor)
                    key = headers[idx] if idx < len(headers) else f"column_{idx}"
                    row_data[key.strip()] = text
                transactions.append(row_data)
    return transactions

def merge_output(document: documentai.Document) -> Dict[str, Any]:
    return {
        "personal_details": extract_key_value_pairs(document),
        "transactions": extract_transactions(document),
        "text_coordinates": extract_text_with_coordinates(document)
    }

if __name__ == "__main__":
    # === Configuration ===
    project_id = "753864940035"
    location = "us"
    processor_id = "f1ffe2360d31483c"
    file_path = "ITR DOC/BANK STATEMENT/BANK.pdf"
    output_path = "output/exp_statement_output.json"
    mime_type = "application/pdf"

    # Process and extract
    doc = process_document(project_id, location, processor_id, file_path, mime_type)
    result = merge_output(doc)

    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    print(f"✅ Output saved to: {output_path}")
