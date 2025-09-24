import re
import numpy as np
from typing import List, Dict, Optional

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
    def repl(_match):
        code = _match.group(0)
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
            # IMPORTANT: keep key terms (title, categories, locations, body) in the text we embed/filter on
            text = (
                f"Type: {ntype}\n"
                f"Title: {title}\n"
                f"Categories: {categories}\n"
                f"Locations: {locations}\n"
                f"Language: {lang}\n"
                f"Content: {body}"
            )
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
    if not gate or len(gate) < 1:
        return "Unknown"
    letter = gate[0].upper()
    return {
        "A": "Concourse A",
        "B": "Concourse B",
        "C": "Concourse C",
        "D": "Concourse D",
        "E": "Concourse E"
    }.get(letter, "Unknown")

def _filter_by_cuisine(chunks: List[Dict], cuisine: str) -> List[Dict]:
    if not cuisine:
        return []
    c = cuisine.lower()
    out = []
    for item in chunks:
        t = item["text"].lower()
        # match cuisine word in title/categories/content
        if c in t:
            out.append(item)
    return out

def search_similar(
    client,
    model,
    question: str,
    embedded_chunks: List[Dict],
    gate: str = "",
    cuisine: Optional[str] = None
) -> str:
    """Optionally prioritizes cuisine matches; falls back to concourse, then global."""
    # Embed question
    res = client.embeddings.create(model=model, input=question)
    q_embedding = res.data[0].embedding

    # Start with a cuisine filter if provided
    candidates: List[Dict] = []
    if cuisine:
        cuisine_hits = _filter_by_cuisine(embedded_chunks, cuisine)
        if cuisine_hits:
            candidates = cuisine_hits

    # If no cuisine hits, use concourse filter (based on gate)
    if not candidates:
        concourse = infer_concourse_from_gate(gate)
        if concourse != "Unknown":
            candidates = [ch for ch in embedded_chunks if concourse.lower() in ch["text"].lower()]

    # If still empty, use all
    if not candidates:
        candidates = embedded_chunks

    # Rank by cosine similarity
    sims = []
    for item in candidates:
        sims.append((cosine_similarity(q_embedding, item["embedding"]), item["text"]))
    sims.sort(reverse=True)

    # Return top 3 joined as context
    top = [t for _, t in sims[:3]]
    return "\n\n".join(top)
