from flask import Flask, request, render_template
import os
import json
from dotenv import load_dotenv
from openai import AzureOpenAI
from smart import format_chunks, embed_chunks, search_similar, generate_answer

load_dotenv()

app = Flask(__name__)

# Azure OpenAI configuration
client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2023-12-01-preview"
)

EMBED_MODEL = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
CHAT_MODEL = os.getenv("AZURE_CHAT_DEPLOYMENT")

# Load and embed data once on startup
with open("data/airport_data.json", "r") as f:
    data = json.load(f)

chunks = format_chunks(data)
embedded_chunks = embed_chunks(client, EMBED_MODEL, chunks)

@app.route("/", methods=["GET", "POST"])
def index():
    answer = ""
    question = ""
    if request.method == "POST":
        question = request.form.get("question")
        if question:
            context = search_similar(client, EMBED_MODEL, question, embedded_chunks)
            answer = generate_answer(client, CHAT_MODEL, context, question)
    return render_template("index.html", answer=answer, question=question)

if __name__ == "__main__":
    app.run(debug=True)
