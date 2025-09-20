import json
import re
import os
from openai import AzureOpenAI
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2023-12-01-preview"
)

EMBED_MODEL = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
CHAT_MODEL = os.getenv("AZURE_CHAT_DEPLOYMENT")


# ---------- Human-readable Location ----------
def parse_location_code(code: str) -> str:
    parts = code.split("-")
    if len(parts) != 3:
        return code  # Return as-is if format is unexpected

    # Terminal (B01 = HIA, we assume always HIA here)
    terminal = "HIA"

    # Level
    level_code = parts[1]  # e.g., "UL001"
    level = level_code[4]  # "1"
    level_str = f"Level {level}"

    # Area
    area_code = parts[2]  # e.g., "IDA0431"
    area_letter = area_code[2]  # A/B/C etc.
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


# ---------- Clean HTML ----------
def clean_html(raw_html: str) -> str:
    if not raw_html:
        return ""
    cleanr = re.compile('<.*?>')
    return re.sub(cleanr, '', raw_html)

# ---------- Format JSON Chunks ----------

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

def replace_location_codes(text: str) -> str:
    pattern = r'B01-UL\d{3}-ID([A-Z])\d{4}'

    def repl(match):
        code = match.group(0)
        return parse_location_code(code)

    return re.sub(pattern, repl, text)


# ---------- Embed Chunks ----------
def embed_chunks(chunks: List[str]) -> List[Dict]:
    embeddings = []
    for i, chunk in enumerate(chunks):
        res = client.embeddings.create(
            model=EMBED_MODEL,
            input=chunk
        )
        embedding = res.data[0].embedding
        embeddings.append({"text": chunk, "embedding": embedding})
    return embeddings

# ---------- Similarity Search ----------
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search_similar(question: str, embedded_chunks: List[Dict]) -> str:
    res = client.embeddings.create(model=EMBED_MODEL, input=question)
    q_embedding = res.data[0].embedding

    similarities = []
    for item in embedded_chunks:
        score = cosine_similarity(q_embedding, item['embedding'])
        similarities.append((score, item['text']))

    similarities.sort(reverse=True)
    top_results = [text for _, text in similarities[:3]]
    return "\n\n".join(top_results)

# ---------- Generate Answer ----------
def generate_answer(context: str, question: str) -> str:
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful airport assistant that recommends places for shopping, dining, and relaxing."},
            {"role": "user", "content": f"Context:\n{context}"},
            {"role": "user", "content": f"Question:\n{question}"}
        ],
        temperature=0.7,
        max_tokens=500
    )
    return response.choices[0].message.content

# ---------- Main Loop ----------
def main():
    with open("data/airport_data.json", "r") as f:
        data = json.load(f)

    chunks = format_chunks(data)
    embedded_chunks = embed_chunks(chunks)

    print("Ask your airport question (or type 'exit'):")
    while True:
        q = input(">> ")
        if q.lower() in ["exit", "quit"]:
            break
        context = search_similar(q, embedded_chunks)
        answer = generate_answer(context, q)
        print("\n--- Answer ---")
        print(answer)
        print("\n")

if __name__ == "__main__":
    main()
