import json
import os
from PIL import Image, ImageDraw
from pdf2image import convert_from_path
import re

def parse_bbox_string(bbox_str):
    # Convert string "(x0, y0, x1, y1)" to list of 4 corner dicts
    match = re.match(r"\((.*?), (.*?), (.*?), (.*?)\)", bbox_str)
    if match:
        x0, y0, x1, y1 = map(float, match.groups())
        return [
            {"x": x0, "y": y0},
            {"x": x1, "y": y0},
            {"x": x1, "y": y1},
            {"x": x0, "y": y1}
        ]
    return []

def get_base_size_from_boxes(items):
    max_x, max_y = 0, 0
    for item in items:
        for pt in item["bbox"]:
            x, y = pt.get("x", 0), pt.get("y", 0)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
    return max_x, max_y

def draw_boxes_on_image_object(img, items, base_width, base_height, color_map):
    img_width, img_height = img.size
    scale_x = img_width / base_width if base_width > 0 else 1
    scale_y = img_height / base_height if base_height > 0 else 1

    draw = ImageDraw.Draw(img)

    for item in items:
        bbox = item["bbox"]
        label_type = item["type"]
        try:
            points = [(pt["x"] * scale_x, pt["y"] * scale_y) for pt in bbox]
            color = color_map.get(label_type, "black")
            draw.polygon(points, outline=color, width=2)
        except Exception as e:
            print(f"⚠️ Skipping invalid bbox: {bbox}, error: {e}")
            continue
    return img

def draw_boxes_on_file(json_path, input_path, output_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    color_map = {
        "account_holder_name": "blue",
        "account_number": "green",
        "address": "purple",
        "text": "red"
    }

    all_items = []
    seen_bboxes = set()  # Keep track of (page, bbox_tuple)

    # Extract bounding boxes from personal_details
    for item in data.get("personal_details", []):
        bbox = item.get("bounding_box", [])
        if bbox:
            bbox_tuple = tuple((pt['x'], pt['y']) for pt in bbox)
            key = (1, bbox_tuple)  # Assume personal details are on page 1
            if key not in seen_bboxes:
                label_type = item.get("type", "text")
                all_items.append({"bbox": bbox, "type": label_type, "page": 1})
                seen_bboxes.add(key)

    # Extract from text_coordinates
    for item in data.get("text_coordinates", []):
        bbox_str = item.get("bbox")
        page = int(item.get("page", 1))
        bbox = parse_bbox_string(bbox_str)
        if bbox:
            bbox_tuple = tuple((pt['x'], pt['y']) for pt in bbox)
            key = (page, bbox_tuple)
            if key not in seen_bboxes:
                all_items.append({"bbox": bbox, "type": "text", "page": page})
                seen_bboxes.add(key)

    if not all_items:
        print("❌ No valid bounding boxes found.")
        return

    base_width, base_height = get_base_size_from_boxes(all_items)

    ext = os.path.splitext(input_path)[-1].lower()
    output_images = []

    if ext == ".pdf":
        print("🔄 Processing scanned PDF pages...")
        images = convert_from_path(input_path, dpi=300)

        for i, page_img in enumerate(images, start=1):
            page_items = [item for item in all_items if item.get("page", 1) == i]
            img_annotated = draw_boxes_on_image_object(
                page_img.copy(), page_items, base_width, base_height, color_map
            )
            output_images.append(img_annotated)

        output_images[0].save(output_path, save_all=True, append_images=output_images[1:])
        print(f"✅ Annotated scanned PDF saved to: {output_path}")
    else:
        img = Image.open(input_path).convert("RGB")
        img_annotated = draw_boxes_on_image_object(img, all_items, base_width, base_height, color_map)
        img_annotated.save(output_path)
        print(f"✅ Annotated image saved to: {output_path}")

# === Usage ===
json_file = "experiment_extracted_data2.json"
input_file = "ITR DOC/BANK STATEMENT/AXIS BANK STATEMENT.pdf"
output_file = "annotated_output.pdf"
draw_boxes_on_file(json_file, input_file, output_file)
