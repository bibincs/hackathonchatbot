# --- Revised Python Script with Enhanced Recommendation Logic ---

from openai import AzureOpenAI
from dotenv import load_dotenv
import os
import json
import re
from flask import Flask, render_template, request, jsonify

# --- Step 1: Load environment variables and set up Azure OpenAI client ---
load_dotenv()
client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2023-12-01-preview"
)
CHAT_MODEL = os.getenv("AZURE_CHAT_DEPLOYMENT")

# --- Step 2: Flask App Setup ---
app = Flask(__name__)

# --- Step 3: Load Airport Data and define Keywords ---
AIRPORT_DATA_FILE = "data/catalog.jsonl"
AIRPORT_DATA = []
POPULAR_ITEMS = {
    "shop": [],
    "dine": [],
    "relax": []
}

# Updated keywords to include more variants for better matching
shop_keywords = ['shop', 'store', 'boutique', 'retail', 'perfume', 'electronics', 'gifts', 'duty free','watches','clothes','apparel','shoes','jewelry','bags','cosmetics','accessories','fashion','toys','books','souvenirs','travel essentials','health','beauty','luxury','convenience','pharmacy','newsstand','magazine','liquor','candy','snacks','tech','gadgets','home decor','art','crafts','handicrafts','local specialties']
dine_keywords = ['restaurant', 'cafe', 'bistro', 'dine', 'food', 'bar', 'lounge', 'food court', 'coffee', 'snack','asisn','italian','mexican','american','indian','chinese','japanese','thai','mediterranean','vegetarian','vegan','gluten-free','desserts','pastries','bakery','fast food','grill','seafood','steakhouse','brunch','buffet','cocktails','wine','beer']
relax_keywords = ['sleep', 'prayer', 'lounge', 'spa', 'rest', 'massages', 'relax', 'quiet','meditation','yoga','wellness','fitness','shower','nap','tranquility','calm','serenity','rejuvenate','unwind','comfort','therapy','health club','sauna','steam room','hot tub','pool','reading room','business center']


def load_airport_data():
    """Loads airport data from the JSONL file."""
    global AIRPORT_DATA
    AIRPORT_DATA = []
    if os.path.exists(AIRPORT_DATA_FILE):
        try:
            with open(AIRPORT_DATA_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    stripped_line = line.strip()
                    if stripped_line:
                        try:
                            AIRPORT_DATA.append(json.loads(stripped_line))
                        except json.JSONDecodeError as e:
                            print(f"Error decoding JSON object from line: {e}")
                            print(f"Problematic line: {stripped_line}")
            print(f"Successfully loaded {len(AIRPORT_DATA)} items from {AIRPORT_DATA_FILE}")
        except Exception as e:
            print(f"Error loading {AIRPORT_DATA_FILE}: {e}")
            AIRPORT_DATA = []
    else:
        print(f"Warning: {AIRPORT_DATA_FILE} not found. Recommendations will not be airport-specific.")

def get_location_name(location_data):
    """Converts location concourse and floor data into a human-readable name."""
    concourse = location_data.get('concourse', '')
    floor = location_data.get('floor', '')
    
    if concourse and floor:
        # Check if the concourse and floor are not null or empty
        if concourse.strip() and floor.strip():
            return f"{concourse}, {floor}"
    elif concourse:
        if concourse.strip():
            return concourse
    elif floor:
        if floor.strip():
            return floor
            
    return "unspecified location"

def get_top_popular_items():
    """Analyzes airport data to find the most popular items in each category."""
    global POPULAR_ITEMS
    category_counts = {
        "shop": {},
        "dine": {},
        "relax": {}
    }
    for item in AIRPORT_DATA:
        entity_type = item.get("entity_type", "").lower()
        if entity_type in category_counts:
            title = item.get("title", "")
            if title:
                category_counts[entity_type][title] = category_counts[entity_type].get(title, 0) + 1
    
    for category in POPULAR_ITEMS.keys():
        sorted_items = sorted(category_counts[category].items(), key=lambda x: x[1], reverse=True)
        POPULAR_ITEMS[category] = [item[0] for item in sorted_items[:3]]

# --- Step 4: Functions for User Data Persistence ---
USER_DATA_FILE = "user_data.json"

def load_user_data_from_file():
    """Loads user data from a JSON file."""
    if os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_user_data_to_file(data):
    """Saves user data to a JSON file."""
    with open(USER_DATA_FILE, 'w') as f:
        json.dump(data, f, indent=4)

# --- Step 5: New & Improved Recommendation Logic ---
def get_category_from_keywords(item, keywords_map):
    """Determines the category of an item based on its keywords."""
    all_keywords = item.get('keywords', []) + item.get('synonyms', [])
    content = item.get('content', '').lower()
    
    for category, keywords in keywords_map.items():
        for keyword in keywords:
            if keyword in content:
                return category
            
            # Check for keyword matches in the tags
            if any(tag.lower() == keyword for tag in item.get('tags', [])):
                return category
    return 'unspecified'

def filter_recommendations(interests, travel_purpose, country_of_origin):
    """
    Selects the best recommendations based on a scoring system.
    """
    categorized_data = {
        "shop": [],
        "dine": [],
        "relax": []
    }
    
    keywords_map = {
        "shop": shop_keywords,
        "dine": dine_keywords,
        "relax": relax_keywords
    }
    
    # First, categorize items based on both entity_type and new keywords
    for item in AIRPORT_DATA:
        entity_type = item.get("entity_type", "").lower()
        if entity_type in categorized_data:
            categorized_data[entity_type].append(item)
            
        # Use keywords as a fallback or secondary categorization
        keyword_category = get_category_from_keywords(item, keywords_map)
        if keyword_category != 'unspecified' and item not in categorized_data.get(keyword_category, []):
            categorized_data[keyword_category].append(item)


    recommendations = {}
    
    for category, items in categorized_data.items():
        best_item = None
        best_score = -1

        for item in items:
            score = 0
            
            # Factor 1: Interest match (highest priority)
            if interests:
                for interest in interests:
                    if interest.lower() in item.get('content', '').lower() or interest.lower() in item.get('tags', []):
                        score += 5
            
            # Factor 2: Country of Origin match for dining
            if category == 'dine' and country_of_origin and item.get('cuisine'):
                cuisine_list = item['cuisine']
                if any(country_of_origin.lower() in c.lower() for c in cuisine_list):
                    score += 4
                    
            # Factor 3: Specific location provided
            if item.get('concourse'):
                score += 3
            
            # Factor 4: Match travel purpose
            if travel_purpose and travel_purpose.lower() in item.get('description', '').lower() or travel_purpose.lower() in item.get('keywords', []):
                score += 2

            # Factor 5: Popularity (as a tie-breaker or general boost)
            if item.get('title') in POPULAR_ITEMS.get(category, []):
                score += 1
            
            if score > best_score:
                best_score = score
                best_item = item
            
        if best_item:
            recommendations[category] = [best_item]
        elif items:
            recommendations[category] = [items[0]]
        else:
            recommendations[category] = []
    
    return recommendations

# --- Step 6: Function to Generate Recommendations (Revised) ---
def generate_recommendations(passenger):
    """
    Generates personalized recommendations using the Azure OpenAI API and airport data.
    """
    interests = passenger.get('interests', [])
    travel_purpose = passenger.get('travel_purpose', '')
    country_of_origin = passenger.get('country_of_origin', '')
    
    relevant_content = filter_recommendations(interests, travel_purpose, country_of_origin)

    summary_lines = []
    for category, items in relevant_content.items():
        if items:
            item = items[0]
            title = item.get('title', 'N/A')
            location = get_location_name(item)
            summary_line = f"One {category} recommendation is {title} located at {location}. The item's keywords are {', '.join(item.get('keywords', []))}."
            summary_lines.append(summary_line)

    relevant_content_text = "\n".join(summary_lines)
    
    prompt = f"""
    You are a friendly and knowledgeable local travel expert based at the airport. Your goal is to provide a brief, personalized, and engaging greeting and recommendation to a visitor.
    
    You must use the exact titles and locations provided in the recommendations list. Do not add any extra recommendations or information not found in the list.

    Passenger Details:
    - Name: {passenger.get('name', 'N/A')}
    - Country of Origin: {passenger.get('country_of_origin', 'N/A')}
    - Interests: {', '.join(passenger.get('interests', []))}
    - Travel Purpose: {passenger.get('travel_purpose', 'N/A')}
    - Flight Details: Flight {passenger.get('flight_number', 'N/A')} from Gate {passenger.get('gate', 'N/A')} at {passenger.get('time', 'N/A')}.
    
    Here are the recommendations to include in your response:
    {relevant_content_text}

    Instructions for your response:
    1. Start with a friendly greeting to the passenger, referencing their name and flight details.
    2. Briefly mention the recommendations you've identified, using only the titles and locations provided. Make it sound natural and helpful. For example, "Since you're interested in [interest], you might enjoy a visit to [recommendation title] located at [location]."
    3. Conclude with a warm, encouraging closing statement and wish them a safe flight.
    """
    
    greeting = "Hello there! I'm your personal travel assistant, ready to help you make the most of your time at the airport."
    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful travel assistant, providing recommendations from within an airport."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        greeting = response.choices[0].message.content
    except Exception as e:
        print(f"Error generating greeting: {e}")
        greeting = f"Hi {passenger.get('name', 'there')}! I'm your travel assistant. We couldn't get a personalized AI response right now, but here are some recommendations based on your preferences: {relevant_content_text}"

    return {
        "greeting": greeting,
        "recommendations": relevant_content
    }

# --- Step 7: Flask Routes ---
@app.route("/")
def home():
    """Serves the main HTML page for the boarding pass scanner."""
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    """
    Receives JSON data from the web page, generates recommendations,
    and returns them as a JSON response.
    """
    passenger_data = request.get_json()
    name = passenger_data.get('name')

    if not name:
        return jsonify({"error": "Name not provided"}), 400

    profiles = load_user_data_from_file()
    status = 'existing'

    if name in profiles:
        print(f"Existing profile found for {name}. Using previous data.")
        passenger_to_use = profiles[name]
    else:
        print(f"No existing profile found for {name}. Saving new data.")
        profiles[name] = passenger_data
        save_user_data_to_file(profiles)
        passenger_to_use = passenger_data
        status = 'new'

    print("=" * 50)
    print(f"--- Generating recommendations for {passenger_to_use['name']} ---")
    response_data = generate_recommendations(passenger_to_use)
    print("=" * 50)
    
    response_data['status'] = status
    
    return jsonify(response_data)


# --- Step 8: Main Execution Loop ---
if __name__ == "__main__":
    load_airport_data()
    get_top_popular_items()
    print("Starting the Travel Recommendation Engine web server...\n")
    app.run(debug=True)