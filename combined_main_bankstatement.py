import os, json, fitz, spacy, pdfplumber, re
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from google.cloud import documentai_v1beta3 as documentai
from google.oauth2 import service_account
from google import genai

# ─── CONFIG ───
PROJECT_ID = "753864940035"
LOCATION = "us"
PROCESSOR_ID = "f1ffe2360d31483c"
CREDENTIAL_PATH = "vision-454216-e49450484a5a.json"
PROCESSOR_NAME = f"projects/{PROJECT_ID}/locations/{LOCATION}/processors/{PROCESSOR_ID}"

# Gemini API Config
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
    label: str
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

# ─── Utils ───
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
    return [NamedEntity(text=ent.text, label=ent.label_, confidence=1.0) for ent in doc.ents]

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

                # Detect if the first row is a header
                is_header = all(cell is not None and not cell.strip().isdigit() for cell in first_row)

                if is_header:
                    headers = [cell.strip() if cell else f"Column_{i}" for i, cell in enumerate(first_row)]
                    rows = table[1:]
                    persistent_headers = headers  # store headers for continuity
                elif persistent_headers:
                    headers = persistent_headers
                    rows = table
                else:
                    # fallback to dynamic headers
                    headers = [f"Column_{i}" for i in range(len(first_row))]
                    rows = table[1:]

                for row in rows:
                    row_dict = {
                        headers[i]: row[i].replace("\n", " ").strip() if row[i] else ""
                        for i in range(min(len(headers), len(row)))
                    }
                    transactions.append(row_dict)

    return transactions

# New function for extracting personal details using Gemini
def extract_personal_details_with_gemini(pdf_path: str) -> dict:
    try:
        # Initialize Gemini API client
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        # Upload file to Gemini
        uploaded_file = client.files.upload(
            file=pdf_path, 
            config={'display_name': os.path.basename(pdf_path)}
        )
        
        # Prompt for extracting personal details
        prompt = """
        Extract the following details from this bank statement PDF:
        - account_holder_name
        - account_number
        - address (if available)

        Only return these in the following compact JSON format:
        {
          "account_holder_name": "...",
          "account_number": "...",
          "address": "..."
        }
        """

        # Generate content using Gemini model
        response = client.models.generate_content(
            model=GEMINI_MODEL_ID,
            contents=[prompt, uploaded_file],
            config={'response_mime_type': 'application/json'}
        )

        # Parse the response
        result = json.loads(response.text)
        
        if isinstance(result, list) and len(result) > 0:
            return result[0]
        elif isinstance(result, dict):
            return result
        else:
            print("⚠️ Unexpected format for personal info. Expected dict, got:", type(result))
            return {}
            
    except Exception as e:
        print(f"❌ Error extracting personal details with Gemini: {e}")
        return {}

# ─── Main Orchestrator ───
def extract_bank_statement(pdf_path: str, output_json_path: str):
    if not os.path.exists(pdf_path):
        print("❌ File not found")
        return

    print("📍 Extracting text with coordinates...")
    text_coords = extract_text_with_coordinates(pdf_path)
    full_text = " ".join([t.text for t in text_coords])

    print("🧠 Named Entities...")
    named_ents = extract_named_entities(full_text)

    print("🔐 Document AI (KVs + Tables)...")
    kvs, table_cells, personal_details_docai = extract_with_docai(pdf_path)

    print("🧑 Extracting personal details with Gemini...")
    gemini_details = extract_personal_details_with_gemini(pdf_path)
    
    # Convert Gemini details to PersonalDetail objects
    personal_details_gemini = []
    for key, value in gemini_details.items():
        if value and value.strip():  # Skip empty values
            personal_details_gemini.append(
                PersonalDetail(
                    type=key,
                    value=value,
                    confidence=0.95,  # High confidence for LLM extractions
                )
            )
    
    # Combine both personal detail extraction methods, prioritizing Gemini
    personal_details = personal_details_gemini
    
    # Add DocAI personal details only if they don't overlap with Gemini ones
    gemini_types = {detail.type for detail in personal_details_gemini}
    for detail in personal_details_docai:
        if detail.type not in gemini_types:
            personal_details.append(detail)

    print("📊 Tables (backup via pdfplumber)...")
    transactions = extract_tables_with_pdfplumber(pdf_path)

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
    print(f"✅ Extraction saved to {output_json_path}")

# ─── Entry Point ───
if __name__ == "__main__":
    extract_bank_statement(
        pdf_path="ITR DOC\BANK STATEMENT\AXIS BANK STATEMENT.pdf",
        output_json_path="final_extracted_data.json"
    )