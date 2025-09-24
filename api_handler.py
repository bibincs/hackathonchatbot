# Install Flask: pip install Flask
# Install the OpenAI client library: pip install openai

import os
import json
from openai import AzureOpenAI
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# --- IMPORTANT: Configure your Azure OpenAI Credentials ---
# Set the following environment variables for security.
# You can find these in your Azure portal under your OpenAI resource.
# AZURE_OPENAI_KEY: Your API key
# AZURE_OPENAI_ENDPOINT: Your endpoint URL (e.g., https://<your-resource-name>.openai.azure.com/)
# AZURE_OPENAI_DEPLOYMENT_NAME: The name of your deployed model (e.g., 'gpt-35-turbo')
# Example:
# Linux/macOS:
# export AZURE_OPENAI_KEY="YOUR_KEY"
# export AZURE_OPENAI_ENDPOINT="YOUR_ENDPOINT"
# export AZURE_OPENAI_DEPLOYMENT_NAME="YOUR_DEPLOYMENT_NAME"
# Windows:
# set AZURE_OPENAI_KEY="YOUR_KEY"
# set AZURE_OPENAI_ENDPOINT="YOUR_ENDPOINT"
# set AZURE_OPENAI_DEPLOYMENT_NAME="YOUR_DEPLOYMENT_NAME"

load_dotenv()

AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_CHAT_DEPLOYMENT")

# Validate environment variables
if not all([AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT_NAME]):
    raise ValueError("Azure OpenAI environment variables not found. Please set them.")

# Initialize the Azure OpenAI client
client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version="2024-02-15-preview" # Use a modern API version
)

app = Flask(__name__)
# Enable CORS to allow the HTML file to make requests to this server
CORS(app)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "supersecret")


@app.route('/generate-itinerary', methods=['POST'])
def generate_itinerary():
    """
    Handles POST requests from the frontend to generate a travel itinerary
    using the Azure OpenAI API.
    """
    try:
        data = request.json
        prompt_from_frontend = data.get('prompt')

        if not prompt_from_frontend:
            return jsonify({"error": "No prompt provided"}), 400

        # Create the full prompt for the model
        full_prompt = f"""
            Act as a travel expert. Generate a single, highly-curated point of interest based on the following user input:
            {prompt_from_frontend}
            
            The response must be a JSON object with the following schema:
            {{
              "name": "string",
              "description": "string",
              "lat": "number",
              "lon": "number",
              "image": "string"
            }}
        """

        # Make the API call to Azure OpenAI's chat completion endpoint
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are a travel expert who provides JSON responses."},
                {"role": "user", "content": full_prompt}
            ],
            response_format={"type": "json_object"}
        )

        # Extract the JSON string from the response and parse it
        response_text = response.choices[0].message.content
        return jsonify(json.loads(response_text)), 200

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "An error occurred while generating the itinerary. Please check your Azure credentials and model deployment."}), 500

if __name__ == '__main__':
    # The server will run on http://127.0.0.1:5000
    app.run(debug=True)
