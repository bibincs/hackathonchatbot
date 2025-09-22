import re
import numpy as np
from typing import List, Dict

def parse_location_code(code: str) -> str:
    parts = code.split("-")
    if len(parts) != 3:
        return code
    level_code = parts[1]
    level = level_code[4]
    level_str = f"Level {level}"
    area_code = parts[2]
    area_letter = area_code[2]
    concourse_map = {
        "A": "Concourse A",
        "B": "Concourse B",
        "C": "Concourse C",
        "D": "Concourse D",
        "E": "Concourse E",
        "L": "Landside",
        "U": "Unknown Area"
    }
    concourse = concourse_map.get(area_letter.upper(), f"Area {area_letter}")
    return f"{concourse}, {level_str}"

def clean_html(raw_html: str) -> str:
    if not raw_html:
        return ""
    cleanr = re.compile('<.*?>')
    return re.sub(cleanr, '', raw_html)

def replace_location_codes(text: str) -> str:
    pattern = r'B01-UL\d{3}-ID([A-Z])\d{4}'
    def repl(match):
        code = match.group(0)
        return parse_location_code(code)
    return re.sub(pattern, repl, text)

def format_chunks(data: List[Dict]) -> List[str]:
    chunks = []
    for entry in data:
        ntype = entry.get("mcn_ntype", "")
        raw_locations = entry.get("mcn_map_location", [])
        locations = ", ".join([parse_location_code(loc) for loc in raw_locations]) or "No location info"
        categories = ", ".join(entry.get("mcn_category", [])) or "No categories"
        for content in entry.get("mcn_content", []):
            title = content.get("mcn_title", "").strip()
            body_raw = clean_html(content.get("mcn_body", "")).strip()
            body = replace_location_codes(body_raw)
            lang = content.get("mcn_language", "en")
            text = f"Type: {ntype}\nTitle: {title}\nCategories: {categories}\nLocations: {locations}\nLanguage: {lang}\nContent: {body}"
            chunks.append(text)
    return chunks

def embed_chunks(client, model, chunks: List[str]) -> List[Dict]:
    embeddings = []
    for chunk in chunks:
        res = client.embeddings.create(model=model, input=chunk)
        embedding = res.data[0].embedding
        embeddings.append({"text": chunk, "embedding": embedding})
    return embeddings

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def infer_concourse_from_gate(gate: str) -> str:
    if not gate or len(gate) < 2:
        return "Unknown"
    letter = gate[0].upper()
    return {
        "A": "Concourse A",
        "B": "Concourse B",
        "C": "Concourse C",
        "D": "Concourse D",
        "E": "Concourse E"
    }.get(letter, "Unknown")

def search_similar(client, model, question: str, embedded_chunks: List[Dict], gate: str = "") -> str:
    res = client.embeddings.create(model=model, input=question)
    q_embedding = res.data[0].embedding

    concourse = infer_concourse_from_gate(gate)

    if concourse != "Unknown":
        filtered_chunks = [chunk for chunk in embedded_chunks if concourse in chunk['text']]
    else:
        filtered_chunks = embedded_chunks

    similarities = [
        (cosine_similarity(q_embedding, item['embedding']), item['text'])
        for item in filtered_chunks
    ]
    similarities.sort(reverse=True)
    top_results = [text for _, text in similarities[:3]]
    return "\n\n".join(top_results)
