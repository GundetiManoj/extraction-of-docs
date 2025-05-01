import os
import json
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from typing import List, Dict, Optional
from google import genai
from pydantic import BaseModel, ValidationError
# ──────── Set Up Gemini API ─────-------
api_key = "AIzaSyAxv3tpH8ZGdLMe6n8kseFDl2QxSGtan9M"  # Replace with your actual key
client= genai.Client(api_key=api_key)
model_id= "gemini-2.0-flash"

# ──────── Pydantic Schema ────────
class BankStatementData(BaseModel):
    account_holder_name: str
    account_number: str
    address: Optional[str]
    transactions: Optional[List[Dict[str, str]]]

# ──────── Personal Info Extraction ────────
def extract_personal_details(pdf_path: str) -> dict:
    uploaded_file = client.files.upload(file=pdf_path, config={'display_name': os.path.basename(pdf_path)})
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

    response = client.models.generate_content(
        model=model_id,
        contents=[prompt, uploaded_file],
        config={'response_mime_type': 'application/json'}
    )

    try:
        result = json.loads(response.text)

        if isinstance(result, list) and len(result) >0:
            return result[0]
        elif isinstance(result, dict):
            return result
        else:
            print("⚠️ Unexpected format for personal info. Expected dict, got:", type(result))
            return {}
    except Exception as e:
        print("❌ Failed to extract personal info:", e)
        return {}


# ──────── Table Extraction ────────
def extract_tables_with_pdfplumber(pdf_path: str) -> List[Dict[str, str]]:
    transactions = []
    persistent_headers = None  # store headers from first table

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                if not table or len(table) < 2:
                    continue  # skip empty or invalid tables

                candidate_headers = [h.strip() if h else "" for h in table[0]]
                is_data_row_like = lambda row: any(cell for cell in row if cell and any(c.isdigit() for c in cell))

                # Use first table's headers persistently
                if persistent_headers is None or candidate_headers != persistent_headers:
                    if is_data_row_like(candidate_headers):  # this isn't actually a header row
                        rows = table
                    else:
                        persistent_headers = candidate_headers
                        rows = table[1:]
                else:
                    rows = table[1:]

                for row in rows:
                    row_dict = {
                        persistent_headers[i]: (row[i].replace('\n', ' ').strip() if row[i] else "")
                        for i in range(min(len(persistent_headers), len(row)))
                        if persistent_headers[i]
                    }
                    transactions.append(row_dict)

    return transactions

# ──────── Fallback OCR Extraction (text only, optional) ────────
def fallback_ocr_extraction(pdf_path: str) -> List[str]:
    print("Fallback: OCRing PDF to extract raw text...")
    images = convert_from_path(pdf_path)
    texts = [pytesseract.image_to_string(img) for img in images]
    with open("ocr_fallback_text.txt", "w", encoding="utf-8") as f:
        f.write("\n\n".join(texts))
    return texts

# ──────── Main Extraction Pipeline ────────
def extract_bank_statement(pdf_path: str, output_json_path: str):
    if not os.path.exists(pdf_path):
        print("❌ File not found.")
        return

    print("📄 Extracting personal details...")
    personal_info = extract_personal_details(pdf_path)

    print("📊 Extracting table data...")
    transactions = extract_tables_with_pdfplumber(pdf_path)

    if not transactions:
        print("⚠️ No tables found, trying OCR fallback.")
        fallback_ocr_extraction(pdf_path)
        print("OCR fallback done. Manual table extraction might be needed.")

    # Final combined structure
    final_data = personal_info.copy() if isinstance(personal_info, dict) else {}
    final_data["transactions"] = transactions if transactions else []


    try:
        # Validate & serialize using Pydantic
        validated = BankStatementData(**final_data)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(validated.model_dump(), f, indent=4)
        print(f"✅ Extraction complete! Saved to: {output_json_path}")
    except ValidationError as ve:
        print("❌ Validation error:", ve)
        return


# ──────── Run ────────
if __name__ == "__main__":
    pdf_path = "ITR DOC/BANK STATEMENT/AXIS BANK STATEMENT.pdf"
    output_json_path = "final_bank_statement_data.json"
    extract_bank_statement(pdf_path, output_json_path)
