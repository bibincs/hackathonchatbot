from flask import Flask

app = Flask(__name__)

import os
import json
import faiss
import numpy as np
from dotenv import load_dotenv
from typing import List
from openai import AzureOpenAI

from bs4 import BeautifulSoup
import re
from typing import List

# Load Azure OpenAI credentials
load_dotenv()

client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2023-12-01-preview"
)

EMBED_MODEL = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
CHAT_MODEL = os.getenv("AZURE_CHAT_DEPLOYMENT")

# 1. Load JSON from local file
def load_airport_data(filepath: str):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERROR] Could not load local data: {e}")
        return []

# 2. Format into text chunks
def clean_html(raw_html):
    if not raw_html:
        return ""
    cleanr = re.compile('<.*?>')
    return re.sub(cleanr, '', raw_html)

def format_chunks(data):
    chunks = []

    if isinstance(data, dict):
        data = [data]

    for item in data:
        item_type = item.get("mcn_ntype", "")
        category = ", ".join(item.get("mcn_category", []))
        location = ", ".join(item.get("mcn_map_location", []))
        contents = item.get("mcn_content", [])

        for content in contents:
            title = clean_html(content.get("mcn_title", "") or "").strip()
            body = clean_html(content.get("mcn_body", "") or "").strip()

            chunk = f"""
Type: {item_type}
Title: {title}
Categories: {category}
Locations: {location}
Content: {body}
            """.strip()

            chunks.append(chunk)

    return chunks

# 3. Embed using Azure OpenAI
def embed_chunks(chunks: List[str]):
    print("[INFO] Embedding chunks...")
    embeddings = []
    for chunk in chunks:
        res = client.embeddings.create(
            input=chunk,
            model=EMBED_MODEL
        )
        embedding = np.array(res.data[0].embedding, dtype="float32")
        embeddings.append(embedding)
    return np.array(embeddings)

# 4. Build FAISS index
def build_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

# 5. Retrieve top-k relevant chunks
def retrieve(query, chunks, index, k=3):
    res = client.embeddings.create(
        input=query,
        model=EMBED_MODEL
    )
    query_vector = np.array(res.data[0].embedding, dtype="float32").reshape(1, -1)
    distances, indices = index.search(query_vector, k)
    return [chunks[i] for i in indices[0]]

# 6. Generate answer with Azure OpenAI
def generate_answer(query, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = f"""
You are a helpful assistant for airport-related queries. Use the airport information below to answer the user's question.

Context:
{context}

Question:
{query}

Answer:
"""

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "You are an airport travel expert."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=500
    )

    return response.choices[0].message.content.strip()

# 7. Main loop
def main():
    print("[INFO] Loading local airport data...")
    data = load_airport_data("data/airport_data.json")
    if not data:
        return

    chunks = format_chunks(data)
    embeddings = embed_chunks(chunks)
    index = build_index(embeddings)

    while True:
        query = input("\nAsk your airport question (or type 'exit'):\n>> ").strip()
        if query.lower() in ["exit", "quit"]:
            break

        context = retrieve(query, chunks, index)
        answer = generate_answer(query, context)

        print("\n--- Answer ---")
        print(answer)

if __name__ == "__main__":
    main()
