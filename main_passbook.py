import os
import json
import pdfplumber
from typing import List, Dict, Optional
from pydantic import BaseModel, ValidationError

# ──────── Pydantic Schema ────────
class EPFContribution(BaseModel):
    wage_month: str
    particulars: str
    date_of_credit: Optional[str]
    epf_wages: Optional[str]
    eps_wages: Optional[str]
    employee_share: Optional[str]
    employer_share: Optional[str]
    pension_contribution: Optional[str]


class EPFPassbookData(BaseModel):
    member_name: str
    member_id: str
    uan: str
    dob: str
    establishment_name: str
    establishment_id: str
    contributions: List[EPFContribution]


# ──────── Extract Personal Info ────────
def extract_personal_details(text: str) -> Dict[str, str]:
    lines = text.splitlines()
    details = {
        "member_name": "",
        "member_id": "",
        "uan": "",
        "dob": "",
        "establishment_name": "",
        "establishment_id": ""
    }
    
    for line in lines:
        if "Member ID/Name" in line:
            parts = line.split("/")
            if len(parts) >= 2:
                details["member_id"] = parts[0].split("|")[-1].strip()
                details["member_name"] = parts[1].strip()
        elif "Establishment ID/Name" in line:
            parts = line.split("/")
            if len(parts) >= 2:
                details["establishment_id"] = parts[0].split("|")[-1].strip()
                details["establishment_name"] = parts[1].strip()
        elif "DOB" in line:
            details["dob"] = line.split("|")[-1].strip()
        elif "UAN" in line:
            details["uan"] = line.split("|")[-1].strip()

    return details


# ──────── Extract Tables ────────
# List of keywords that identify header or irrelevant rows
HEADER_KEYWORDS = [
    "wage month", "particulars", "date of credit", "epf wages",
    "eps wages", "pension", "employee share", "employer share", "opening balance",
    "closing balance", "int.", "interest", "n/a", "osru ekg", "dezpkjh", "vu'knku",
    "fu;ksdrk", "deZpkjh"
]

def is_valid_contribution(row: List[str]) -> bool:
    """Check if a row represents a valid contribution entry."""
    # Skip rows with header keywords
    if any(any(keyword in cell.lower() for keyword in HEADER_KEYWORDS) for cell in row if cell):
        return False
    
    # Skip empty rows or rows with insufficient data
    if len([cell for cell in row if cell.strip()]) < 3:
        return False
    
    # Check if the first cell (wage_month) has a valid month format
    if not row[0] or not any(month in row[0].lower() for month in 
                           ["jan", "feb", "mar", "apr", "may", "jun", 
                            "jul", "aug", "sep", "oct", "nov", "dec"]):
        return False
    
    return True

def extract_contributions(pdf_path: str) -> List[Dict[str, str]]:
    contributions = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            table = page.extract_table()
            if not table or len(table) < 2:
                continue

            # Find the header row index
            header_row_index = -1
            for i, row in enumerate(table):
                if row and any("wage month" in str(cell).lower() for cell in row if cell):
                    header_row_index = i
                    break

            # Process data rows (after header)
            for row in table[header_row_index+1:] if header_row_index >= 0 else table:
                # Clean row data
                row = [str(cell).strip() if cell else "" for cell in row]
                
                # Skip if not a valid contribution row
                if not is_valid_contribution(row):
                    continue
                # Ensure row has enough columns
                if len(row) < 10:
                    row += [""] * (10 - len(row))
                
                contributions.append({
                    "wage_month": row[0],
                    "particulars": row[1],
                    "date_of_credit": row[2],
                    "epf_wages": row[3],
                    "eps_wages": row[4],
                    "employee_share": row[5],
                    "employer_share": row[6],
                    "pension_contribution": row[9]  # Adjusted to match typical column order
                })

    return contributions

# ──────── Main Extraction ────────
def extract_epf_passbook(pdf_path: str, output_json_path: str):
    if not os.path.exists(pdf_path):
        print("File not found.")
        return

    with pdfplumber.open(pdf_path) as pdf:
        all_text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

    print("Extracting personal info...")
    personal_details = extract_personal_details(all_text)

    print("Extracting contributions...")
    contribution_data = extract_contributions(pdf_path)

    # Clean any remaining invalid entries
    cleaned_contributions = []
    for contrib in contribution_data:
        if contrib["wage_month"] and contrib["particulars"]:
            cleaned_contributions.append(contrib)

    data = {**personal_details, "contributions": cleaned_contributions}

    try:
        validated = EPFPassbookData(**data)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(validated.model_dump(), f, indent=4)
        print(f" Extraction complete. Data saved to {output_json_path}")
    except ValidationError as e:
        print(" Validation Error:", e)

# ──────── Run ────────
if __name__ == "__main__":
    pdf_path = "ITR DOC\EMPLOYEE INFO\PF PASSBOOK.pdf"  # Adjust path as needed
    output_json_path = "epf_passbook_data.json"
    extract_epf_passbook(pdf_path, output_json_path)