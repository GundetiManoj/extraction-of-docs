import os, json, fitz, spacy, pdfplumber, re
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from google.cloud import documentai_v1beta3 as documentai
from google.oauth2 import service_account
from google import genai
from line_extraction_bank import online_process, extract_lines_with_layout
import random

# ─── CONFIG ───
PROJECT_ID = "753864940035"
LOCATION = "us"
PROCESSOR_ID = "f1ffe2360d31483c"
CREDENTIAL_PATH = "vision-454216-e49450484a5a.json"
PROCESSOR_NAME = f"projects/{PROJECT_ID}/locations/{LOCATION}/processors/{PROCESSOR_ID}"

GEMINI_API_KEY = "AIzaSyAxv3tpH8ZGdLMe6n8kseFDl2QxSGtan9M"
GEMINI_MODEL_ID = "gemini-2.0-flash"

# ─── Schemas ───
class KVWithConfidence(BaseModel):
    key: str
    value: str
    confidence: float
    page: int
    bounding_box: List[Dict[str, float]]

class TableCell(BaseModel):
    row_index: int
    column_index: int
    text: str
    confidence: float
    page: int
    bounding_box: List[Dict[str, float]]

class PersonalDetail(BaseModel):
    type: str
    value: str
    confidence: float
    page: Optional[int] = 1
    bounding_box: Optional[List[Dict[str, float]]] = []

class TextSpan(BaseModel):
    text: str
    bbox: str
    page: str

class NamedEntity(BaseModel):
    text: str
    confidence: float

class BankStatementSchema(BaseModel):
    account_holder_name: Optional[str] = None
    account_number: Optional[str] = None
    address: Optional[str] = None
    personal_details: List[PersonalDetail]
    transactions: List[Dict[str, str]]
    text_coordinates: List[TextSpan]
    named_entities: List[NamedEntity]
    key_value_pairs: List[KVWithConfidence]
    table_cells: List[TableCell]

# ─── Utility Functions ───
def extract_text_with_coordinates(pdf_path: str) -> List[TextSpan]:
    doc = fitz.open(pdf_path)
    result = []
    for page_num, page in enumerate(doc, start=1):
        for block in page.get_text("dict")["blocks"]:
            if block["type"] == 0:
                for line in block["lines"]:
                    for span in line["spans"]:
                        result.append(TextSpan(
                            text=span["text"],
                            bbox=str(span["bbox"]),
                            page=str(page_num)
                        ))
    return result

def extract_named_entities(text: str) -> List[NamedEntity]:
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return [NamedEntity(text=ent.text, confidence=random.uniform(0.7, 1.0)) for ent in doc.ents] 

def extract_with_docai(pdf_path: str):
    credentials = service_account.Credentials.from_service_account_file(CREDENTIAL_PATH)
    client = documentai.DocumentProcessorServiceClient(credentials=credentials)

    with open(pdf_path, "rb") as f:
        raw_document = documentai.RawDocument(content=f.read(), mime_type="application/pdf")

    request = documentai.ProcessRequest(name=PROCESSOR_NAME, raw_document=raw_document)
    document = client.process_document(request=request).document

    kvs, table_cells, personal_details = [], [], []

    for entity in document.entities:
        page_ref = entity.page_anchor.page_refs[0] if entity.page_anchor.page_refs else None
        box = [{"x": v.x, "y": v.y} for v in page_ref.bounding_poly.vertices] if page_ref else []
        kv = KVWithConfidence(
            key=entity.type_,
            value=entity.mention_text,
            confidence=entity.confidence,
            page=page_ref.page if page_ref else 1,
            bounding_box=box
        )
        kvs.append(kv)
        if any(k in entity.type_.lower() for k in ["name", "account", "address", "phone", "email"]):
            personal_details.append(PersonalDetail(
                type=entity.type_,
                value=entity.mention_text,
                confidence=entity.confidence,
                page=page_ref.page if page_ref else 1,
                bounding_box=box
            ))

    for page in document.pages:
        for table in page.tables:
            for row_index, row in enumerate(table.header_rows + table.body_rows):
                for col_index, cell in enumerate(row.cells):
                    if cell.layout and cell.layout.text_anchor.text_segments:
                        seg = cell.layout.text_anchor.text_segments[0]
                        text = document.text[seg.start_index:seg.end_index]
                        box = [{"x": v.x, "y": v.y} for v in cell.layout.bounding_poly.vertices]
                        table_cells.append(TableCell(
                            row_index=row_index,
                            column_index=col_index,
                            text=text,
                            confidence=cell.layout.confidence,
                            page=page.page_number,
                            bounding_box=box
                        ))

    return kvs, table_cells, personal_details

def extract_tables_with_pdfplumber(pdf_path: str) -> List[Dict[str, str]]:
    transactions = []
    persistent_headers = None

    with pdfplumber.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()
            for table_index, table in enumerate(tables):
                if not table or len(table) < 2:
                    continue

                first_row = table[0]
                second_row = table[1] if len(table) > 1 else None

                is_header = all(cell is not None and not cell.strip().isdigit() for cell in first_row)

                if is_header:
                    headers = [cell.strip() if cell else f"Column_{i}" for i, cell in enumerate(first_row)]
                    rows = table[1:]
                    persistent_headers = headers
                elif persistent_headers:
                    headers = persistent_headers
                    rows = table
                else:
                    headers = [f"Column_{i}" for i in range(len(first_row))]
                    rows = table[1:]

                for row in rows:
                    row_dict = {
                        headers[i]: row[i].replace("\n", " ").strip() if row[i] else ""
                        for i in range(min(len(headers), len(row)))
                    }
                    transactions.append(row_dict)

    return transactions

def extract_personal_details_with_gemini(pdf_path: str) -> dict:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        uploaded_file = client.files.upload(file=pdf_path, config={'display_name': os.path.basename(pdf_path)})
        prompt = """
        Extract the following details from this bank statement PDF:
        - account_holder_name
        - account_number
        - address (if available)
        Return only JSON in this format:
        {
          "account_holder_name": "...",
          "account_number": "...",
          "address": "..."
        }
        """
        response = client.models.generate_content(
            model=GEMINI_MODEL_ID,
            contents=[prompt, uploaded_file],
            config={'response_mime_type': 'application/json'}
        )
        result = json.loads(response.text)
        return result[0] if isinstance(result, list) else result
    except Exception as e:
        print(f"Error extracting personal details with Gemini: {e}")
        return {}

# ─── Main Orchestrator ───
def extract_bank_statement(pdf_path: str, output_json_path: str):
    if not os.path.exists(pdf_path):
        print("File not found")
        return

    print("Extracting text coordinates...")
    text_coords = extract_text_with_coordinates(pdf_path)
    full_text = " ".join([t.text for t in text_coords])

    print("Named Entities...")
    named_ents = extract_named_entities(full_text)

    print("Document AI KVs + Tables...")
    kvs, table_cells, personal_details_docai = extract_with_docai(pdf_path)

    print("Personal details from Gemini...")
    gemini_details = extract_personal_details_with_gemini(pdf_path)
    def match_bbox_from_text_coords(value: str, text_coords: List[TextSpan]):
        for span in text_coords:
         if value.strip().lower() in span.text.strip().lower():
            try:
                x0, y0, x1, y1 = eval(span.bbox)  # safe only if data is trusted
                return [
                    {"x": x0, "y": y0},
                    {"x": x1, "y": y0},
                    {"x": x1, "y": y1},
                    {"x": x0, "y": y1}
                ], int(span.page)
            except Exception:
                return None, None
        return None, None



# After getting Gemini personal details
    personal_details = []
    gemini_types = set()

    for key, value in gemini_details.items():
     if value:
            bbox, page = match_bbox_from_text_coords(value, text_coords)
            personal_details.append(PersonalDetail(
             type=key,
             value=value,
             confidence=0.95,
             bounding_box=bbox if bbox else [],
             page=page if page else 1
        ))
            gemini_types.add(key.lower())


    # personal_details = []
    # gemini_types = set()
    # for key, value in gemini_details.items():
    #     if value:
    #         personal_details.append(PersonalDetail(type=key, value=value, confidence=0.95))
    #         gemini_types.add(key.lower())

    for pd in personal_details_docai:
        if pd.type.lower() not in gemini_types:
            personal_details.append(pd)

    print("Table extraction via pdfplumber...")
    transactions = extract_tables_with_pdfplumber(pdf_path)

    if not transactions:
        print("Fallback: using layout-based line extractor...")
        doc = online_process(PROJECT_ID, LOCATION, PROCESSOR_ID, pdf_path, "application/pdf")
        transactions = extract_lines_with_layout(doc)

    final_data = BankStatementSchema(
        account_holder_name=gemini_details.get("account_holder_name"),
        account_number=gemini_details.get("account_number"),
        address=gemini_details.get("address"),
        personal_details=personal_details,
        transactions=transactions,
        text_coordinates=text_coords,
        named_entities=named_ents,
        key_value_pairs=kvs,
        table_cells=table_cells
    )

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(final_data.dict(), f, indent=4)
    print(f"✔ Extraction complete. Saved to {output_json_path}")

# Entry Point
if __name__ == "__main__":
    extract_bank_statement(
        pdf_path="ITR DOC\BANK STATEMENT\AXIS BANK STATEMENT.pdf",
        output_json_path="experiment_extracted_data2.json"
    )
# ─── END ───   