from flask import Flask, request, render_template, session, redirect, url_for
import os
import json
from dotenv import load_dotenv
from openai import AzureOpenAI
from smart import format_chunks, embed_chunks, search_similar, infer_concourse_from_gate

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "supersecret")

# Azure OpenAI configuration
client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2023-12-01-preview"
)

EMBED_MODEL = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
CHAT_MODEL = os.getenv("AZURE_CHAT_DEPLOYMENT")

# Load and embed data once
with open("data/airport_data.json", "r") as f:
    data = json.load(f)

chunks = format_chunks(data)
embedded_chunks = embed_chunks(client, EMBED_MODEL, chunks)


@app.route("/", methods=["GET", "POST"])
def chat():
    if "history" not in session:
        session["history"] = []

    # Get inputs from form (populated via the modal)
    passenger_name = request.form.get("passenger_name", "").strip() or "Passenger"
    flight_number = request.form.get("flight_number", "").strip() or "Unknown"
    gate = request.form.get("gate", "").strip().upper()
    boarding_time = request.form.get("time", "").strip()

    # Derive time to flight (mocked for now — you can calculate it later)
    time_to_flight = "4 hours"
    destination = "London (LHR)"  # You can extract this from the flight number if needed

    if request.method == "POST":
        question = request.form.get("question", "").strip()

        if question:
            # Save user message to chat history
            session["history"].append({"role": "user", "content": question})

            # Get context (filtering chunks near the gate)
            context = search_similar(client, EMBED_MODEL, question, embedded_chunks, gate=gate)

            # System prompt personalized based on passenger info
            system_prompt = (
                "You are a smart and friendly airport assistant. The passenger has just scanned their boarding pass and is now interacting with you on a touchscreen or mobile device.\n\n"
                f"You already know the following information:\n"
                f"- Passenger name: {passenger_name}\n"
                f"- Flight number: {flight_number}\n"
                f"- Destination: {destination}\n"
                f"- Time until boarding: {time_to_flight}\n"
                f"- Boarding gate: {gate or 'Unknown'}\n\n"

                f"Start the conversation by greeting the user by name and presenting the key flight info, for example:\n"
                f"\"Hi {passenger_name}, your next flight to {destination} departs in {time_to_flight} from Gate {gate or 'Unknown'}. What would you like to do while you're here?\"\n\n"

                "Then, offer them three clear options:\n"
                "1. Shopping 🛍️\n"
                "2. Dining 🍽️\n"
                "3. Relax 😌\n\n"

                "Wait for the user to select or mention one. If the user chooses Dining or expresses interest in food (e.g., 'I'm hungry', 'I want to have dinner', 'munch', 'explore food', etc.), follow up with:\n"
                "\"Got it! What kind of cuisine are you in the mood for?\"\n\n"

                "Once they answer, ask one more follow-up to help narrow down options. Then say:\n"
                "\"I saved your dining preferences! Would you like to do something else or should I generate your itinerary based on this?\"\n\n"
                "Show three buttons/options:\n"
                "1. ✅ Yes, Generate Itinerary\n"
                "2. 🛍️ Continue with Shopping\n"
                "3. 😌 Continue with Relaxing\n\n"

                "If the user selects 'Generate Itinerary', end the conversation and trigger the itinerary UI.\n"
                "If they select Shopping or Relax, go back to gathering preferences in those categories.\n\n"

                "Your goal is to make the experience smooth, proactive, and personal — always guiding the user while respecting their choices.\n"
            )

            # Compose messages
            full_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}"}
            ] + session["history"]

            # Generate response from OpenAI
            response = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=full_messages,
                temperature=0.7,
                max_tokens=500
            )

            assistant_reply = response.choices[0].message.content

            # Save assistant reply to history
            session["history"].append({"role": "assistant", "content": assistant_reply})
            session.modified = True

    return render_template("index.html", history=session.get("history", []))

@app.route("/scanner", methods=["GET"])
def scanner():
    return render_template("scanner.html")

@app.route("/reset", methods=["GET"])
def reset():
    session.clear()
    return redirect(url_for("chat"))


if __name__ == "__main__":
    app.run(debug=True)
