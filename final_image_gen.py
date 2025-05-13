import json
from PIL import Image, ImageDraw, ImageFont

def draw_boxes_on_blank_page(json_path, output_path, page_size=(2480, 3508)):
    with open(json_path, 'r') as f:
        data = json.load(f)

    valid_items = []
    for section in ["personal_details", "key_value_pairs", "named_entities"]:
        if section in data:
            for item in data[section]:
                if isinstance(item, dict) and "bounding_box" in item and "value" in item:
                    valid_items.append({
                        "value": item["value"],
                        "bbox": item["bounding_box"],
                        "page": item.get("page") if isinstance(item.get("page"), int) else 1
                    })

    for section in ["tables"]:
        if section in data:
            for table in data[section]:
                for row in table.get("rows", []):
                    for item in row:
                        if isinstance(item, dict) and "bounding_box" in item and "value" in item:
                            valid_items.append({
                                "value": item["value"],
                                "bbox": item["bounding_box"],
                                "page": item.get("page") if isinstance(item.get("page"), int) else 1
                            })

    # Organize by page
    pages = {}
    for item in valid_items:
        page = item["page"]
        if page is None:
            continue
        pages.setdefault(page, []).append(item)

    if not pages:
        print("No valid pages found.")
        return

    page_images = []

    for page_num in sorted(pages.keys()):
        img = Image.new("RGB", page_size, "white")
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()

        for item in pages[page_num]:
            bbox = item["bbox"]
            text = item["value"]

            # Handle bounding boxes with dicts or tuples
            if not (isinstance(bbox, list) and len(bbox) == 4):
                continue

            try:
                points = [(pt["x"], pt["y"]) if isinstance(pt, dict) else pt for pt in bbox]
                draw.polygon(points, outline="blue", width=2)

                x_text, y_text = points[0]
                draw.text((x_text + 5, y_text + 2), text, fill="black", font=font)
            except Exception as e:
                print(f"Skipping invalid bbox: {bbox}, error: {e}")
                continue

        page_images.append(img)

    if output_path.endswith(".pdf"):
        page_images[0].save(output_path, save_all=True, append_images=page_images[1:])
    else:
        for i, img in enumerate(page_images):
            img.save(f"{output_path}_page_{i+1}.png")

# === Usage ===
json_file = "final_comparison.json"
output_file = "blank_drawn_output.pdf"
draw_boxes_on_blank_page(json_file, output_file)
