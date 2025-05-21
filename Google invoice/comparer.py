import json
import re
from difflib import SequenceMatcher


def similar(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def find_best_match(text, text_coords, threshold=0.50):
    best_match = None
    best_score = 0
    if not text:
        return None
    for item in text_coords:
        item_text = item.get("text")
        if not item_text:
            continue 
        score = similar(text.strip(), item["text"].strip())
        if score > best_score and score >= threshold:
            best_score = score
            best_match = item
    return best_match


def enrich_fields(data):
    text_coords = data.get("text_with_coords", [])
    
    # --- Enrich Personal Details ---
    for key, value in data.get("personal_details", {}).items():
        if not value:
            continue
        match = find_best_match(value, text_coords)
        if match:
            data["personal_details"][key] = {
                "value": value,
                "bounding_box": match["bounding_box"],
                "page_number": match["page_number"]
            }

    # --- Enrich Named Entities ---
    enriched_entities = []
    for entity in data.get("named_entities", []):
        value = entity.get("name") or entity.get("value")
        if not value:
            continue
        match = find_best_match(value, text_coords)
        enriched = {
            **entity,
            "value": value,
        }
        if match:
            enriched["bounding_box"] = match["bounding_box"]
            enriched["page_number"] = match["page_number"]
        enriched_entities.append(enriched)
    data["named_entities"] = enriched_entities

    # --- Enrich Tables ---
    enriched_tables = []
    for table in data.get("tables", []):
        enriched_table = {
            "headers": [],
            "rows": []
        }

        # Headers
        for header in table.get("headers", []):
            match = find_best_match(header, text_coords)
            enriched_table["headers"].append({
                "text": header,
                "bounding_box": match["bounding_box"] if match else None,
                "page_number": match["page_number"] if match else None
            })

        # Rows
        for row in table.get("rows", []):
            enriched_row = []
            for cell in row:
                match = find_best_match(cell, text_coords)
                enriched_row.append({
                    "text": cell,
                    "bounding_box": match["bounding_box"] if match else None,
                    "page_number": match["page_number"] if match else None
                })
            enriched_table["rows"].append(enriched_row)

        enriched_tables.append(enriched_table)
    data["tables"] = enriched_tables

    return data


if __name__ == "__main__":
    input_path = "output_result.json"
    output_path = "enriched_output.json"
    with open(input_path, "r") as f:
        data = json.load(f)

    enriched_data = enrich_fields(data)

    with open(output_path, "w") as f:
        json.dump(enriched_data, f, indent=2)

    print(f"✅ Enriched output written to: {output_path}")