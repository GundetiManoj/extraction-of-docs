# ─── FILE: line_layout_extractor.py ───
from typing import List, Dict, Any
from google.api_core.client_options import ClientOptions
from google.cloud import documentai_v1 as documentai

def online_process(project_id: str, location: str, processor_id: str, file_path: str, mime_type: str) -> documentai.Document:
    client = documentai.DocumentProcessorServiceClient(
        client_options=ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")
    )
    name = client.processor_path(project_id, location, processor_id)
    with open(file_path, "rb") as f:
        raw_document = documentai.RawDocument(content=f.read(), mime_type=mime_type)
    request = documentai.ProcessRequest(name=name, raw_document=raw_document)
    result = client.process_document(request=request)
    return result.document

def extract_lines_with_layout(document: documentai.Document) -> List[Dict[str, Any]]:
    extracted_lines = []
    for page_number, page in enumerate(document.pages, start=1):
        for line in page.lines:
            line_text = "".join(
                document.text[int(seg.start_index):int(seg.end_index)]
                for seg in line.layout.text_anchor.text_segments
            ).strip()

            bbox = [{"x": v.x, "y": v.y} for v in line.layout.bounding_poly.vertices]
            confidence = line.layout.confidence

            extracted_lines.append({
                "page": page_number,
                "text": line_text,
                "confidence": confidence,
                "bounding_box": bbox,
            })
    return extracted_lines
