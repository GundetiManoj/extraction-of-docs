import json
import fitz  # PyMuPDF
from PIL import Image, ImageDraw
import os
def draw_boxes_on_pdf(input_pdf_path, enriched_json_path, output_folder="InterestCertificate_Home"):
    # Load data
    with open(enriched_json_path, "r") as f:
        data = json.load(f)

    text_coords = data.get("text_with_coords", [])
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open PDF
    doc = fitz.open(input_pdf_path)

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        draw = ImageDraw.Draw(img)

        # Draw each bounding box for this page
        for item in text_coords:
            if item["page_number"] != page_num + 1:
                continue
            bbox = item["bounding_box"]
            w, h = img.size

            box = [
                (bbox["x1"] * w, bbox["y1"] * h),
                (bbox["x2"] * w, bbox["y2"] * h),
                (bbox["x3"] * w, bbox["y3"] * h),
                (bbox["x4"] * w, bbox["y4"] * h),
                (bbox["x1"] * w, bbox["y1"] * h)  # Close the polygon
            ]
            draw.line(box, fill="red", width=2)

        # Save the output image
        out_path = os.path.join(output_folder, f"page_SoA_{page_num + 1}.png")
        img.save(out_path)
        print(f"✅ Saved: {out_path}")

    print("✅ All pages processed.")


if __name__ == "__main__":
    input_pdf = r"ITR DOC\HOME LOAN\StatementOfAccount_Home Loan Cover.pdf"
    enriched_json = "enriched_statementofA.json"
    draw_boxes_on_pdf(input_pdf, enriched_json)