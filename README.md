# 🧾 Document AI parser & Identity Extraction Pipeline

This project leverages **Google Document AI** and **Gemini 1.5** to extract structured data from scanned documents like invoices, Aadhaar cards, and bank statements. It also visualizes bounding boxes around detected text using **PyMuPDF** and **Pillow**.

## 📁 Features

*  Extract key-value pairs, personal details, tables, and named entities
*  Support for PDF documents (invoices, ID proofs, statements, etc.)
*  Gemini integration to extract **personal details** from raw text
*  Bounding box visualization on original PDF pages
*  Clean JSON export of enriched document structure

## 🛠 Setup Instructions

### 1. Clone this repo

```bash
git clone https://github.com/GundetiManoj/extraction-of-docs.git
```

### 2. Create a virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt`:

```txt
google-cloud-documentai
google-generativeai
python-dotenv
pillow
PyMuPDF
```

---

## 🔐 Environment Setup

### Google Document AI

1. Create a processor in [Google Cloud Document AI](https://console.cloud.google.com/ai/document-ai)
2. Download the JSON key and set the path in your script:

```python
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "your-key.json"
```

3. Set your processor/project details in `main.py`:

```python
PROJECT_ID = "your-project-id"
LOCATION = "us"
PROCESSOR_ID = "your-processor-id"
```

### Gemini API Key

Create a `.env` file:

```env
GEMINI_API_KEY=your_gemini_api_key
```

Or set it directly:

```bash
export GEMINI_API_KEY=your_gemini_api_key
```

---

## ▶️ Running the Extractor

### Document Analysis

```bash
python main.py
```

It will:

* Process a PDF using Document AI
* Extract personal details using Gemini
* Output a structured JSON file

Edit the `input_path` and `output_path` in `main.py` for your document.

### Visualize Text with Bounding Boxes

```bash
python verifier_image.py
```

* Draws red boxes around each token on a PDF page using `text_with_coords` from output JSON


## 📂 Project Structure

```
main.py               # Core document processing logic
verifier_image.py         # PDF visualization using PyMuPDF
.env                  # Contains GEMINI_API_KEY
ITR DOC/                  # Folder with input PDFs
    ├── BASIC/
    ├── HOME LOAN/
    └── BANK STATEMENT/
ITR /                  # Folder with output JSONs and images
    ├── BASIC/
    ├── HOME LOAN/
    └── BANK STATEMENT/

```


## Example Output

```json
{
  "personal_details": {
    "name": "RAHUL",
    "email": "Rahul@example.com"
  },
  "text_with_coords": [...],
  "key_value_pairs": [...],
  "named_entities": [...],
  "tables": [...]
}
```


## 📌 Notes

* Works best with clean, machine-readable PDFs and OCR pdfs.
* If `key_value_pairs` or `tables` are missing, fallback extraction using Gemini will still give useful personal data.
* Gemini response is parsed and cleaned before loading into the final JSON.

## 📬 Contact

Feel free to raise issues or contact [Manoj Gundeti](https://www.linkedin.com/in/manoj-gundeti25/) for support or improvements!
