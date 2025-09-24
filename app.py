from flask import Flask, request, render_template, session, redirect, url_for, jsonify
import os, json, random, re, urllib.parse
from dotenv import load_dotenv
from openai import AzureOpenAI
from smart import (
    format_chunks,
    embed_chunks,
    search_similar,            # must accept cuisine=
    infer_concourse_from_gate,
    parse_location_code,
    clean_html,
)

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "supersecret")

# -------- Azure OpenAI --------
client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2023-12-01-preview"
)
EMBED_MODEL = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
CHAT_MODEL  = os.getenv("AZURE_CHAT_DEPLOYMENT")

# -------- HIA Map deep link (with fixed src + multi-waypoints) --------
# Example base from your latest note:
# https://visiophptest-.../mapviewer.html?src=B01-UL001-IDA0394&waypoints=ID1&waypoints=ID2
HIA_MAP_BASE = os.getenv(
    "HIA_MAP_BASE",
    "https://visiophptest-c7fpchb0bkfgajd3.swedencentral-01.azurewebsites.net/visioweb/application/mapviewer-uikit/mapviewer.html"
)
HIA_MAP_SRC = os.getenv("HIA_MAP_SRC", "B01-UL001-IDA0394")
HIA_MAP_FALLBACK_DST = os.getenv("HIA_MAP_FALLBACK_DST", "B01-UL001-IDB0364")

@app.route("/directions", methods=["GET"])
def directions():
    """
    Build a map URL like:
      <BASE>?src=<fixed>&waypoints=<id>&waypoints=<id>...

    Accepts:
      - ?waypoints=ID   (repeat param multiple times OR comma-separated)
      - ?dst=ID         (legacy single id)
    Falls back to HIA_MAP_FALLBACK_DST if nothing provided.
    """
    # Gather waypoints from query
    waypoints = request.args.getlist("waypoints") or []
    if not waypoints:
        # Support comma-separated
        csv = request.args.get("waypoints", "")
        if csv:
            waypoints = [w.strip() for w in csv.split(",") if w.strip()]

    # Back-compat: single ?dst=...
    if not waypoints:
        dst = (request.args.get("dst") or "").strip()
        if dst:
            waypoints = [dst]

    # Fallback
    if not waypoints:
        waypoints = [HIA_MAP_FALLBACK_DST]

    # Build URL
    base = HIA_MAP_BASE
    params = [("src", HIA_MAP_SRC)]
    for wp in waypoints:
        params.append(("waypoints", wp))
    query = "&".join(f"{k}={urllib.parse.quote(v)}" for k, v in params)
    return redirect(f"{base}?{query}")

# Optional back-compat: /directions/<id>
@app.route("/directions/<path:location_id>", methods=["GET"])
def directions_compat(location_id):
    return redirect(url_for("directions", waypoints=location_id))

# -------- Load + embed sources --------
def load_jsonl(path):
    items = []
    if not os.path.exists(path):
        return items
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    items.append(json.loads(line))
                except Exception:
                    pass
    return items

airport_data = []
if os.path.exists("data/airport_data.json"):
    try:
        with open("data/airport_data.json", "r", encoding="utf-8") as f:
            airport_data = json.load(f)
    except Exception:
        airport_data = []

catalog_data = load_jsonl("data/catalog.jsonl")

all_chunks = []
if airport_data:
    all_chunks.extend(format_chunks(airport_data))
if catalog_data:
    all_chunks.extend(format_chunks(catalog_data))

embedded_chunks = embed_chunks(client, EMBED_MODEL, all_chunks)

# -------- Catalog index WITH location_ids + content logos --------
def _norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())

def build_catalog_index(catalog):
    """
    Index by normalized EN title -> list[variant dicts], and a flat list for fuzzy scans.
    Keeps: id, all location_ids, description, concourse text, categories, and mcn_content_logo (preferred).
    """
    idx = {}
    all_items = []
    for row in catalog:
        loc_codes = row.get("mcn_map_location", []) or []
        primary_id = loc_codes[0] if loc_codes else None
        concourse_txt = parse_location_code(primary_id) if primary_id else "Location info not available"
        cats = row.get("mcn_category", []) or []

        # row-level fallback image
        row_image = row.get("image") or row.get("thumbnail") or None

        for content in (row.get("mcn_content") or []):
            title = (content.get("mcn_title") or "").strip()
            if not title:
                continue
            lang  = (content.get("mcn_language") or "en").lower()
            if lang != "en":
                continue

            body_raw = content.get("mcn_body") or ""
            description = clean_html(body_raw).strip()

            content_logo = content.get("mcn_content_logo") or content.get("logo") or None
            image = content_logo or row_image

            item = {
                "id": primary_id,
                "location_ids": loc_codes,
                "name": title,
                "description": description,
                "concourse": concourse_txt,
                "categories": cats,
                "image": image,
            }
            key = _norm_name(title)
            idx.setdefault(key, []).append(item)
            all_items.append(item)

    return idx, all_items

CATALOG_INDEX, CATALOG_ALL = build_catalog_index(catalog_data)

def match_catalog_by_name(name: str):
    if not name:
        return []
    return CATALOG_INDEX.get(_norm_name(name), [])

def _concourse_letter_from_label(label: str) -> str:
    m = re.search(r"Concourse\s+([A-E])", (label or ""), flags=re.I)
    return m.group(1).upper() if m else ""

def best_catalog_match(place: str, gate: str = ""):
    """
    Exact normalized title → fuzzy/substring → prefer same concourse letter as gate.
    Returns dict or None.
    """
    if not place:
        return None

    # 1) exact
    exact = match_catalog_by_name(place)
    candidates = list(exact)

    # 2) fuzzy exact-case-insensitive then substring
    pname = place.strip().lower()
    if not candidates:
        for it in CATALOG_ALL:
            if it.get("name", "").lower() == pname:
                candidates.append(it)
        if not candidates:
            for it in CATALOG_ALL:
                if pname in it.get("name", "").lower():
                    candidates.append(it)

    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    # 3) prefer by concourse letter
    gate_concourse = infer_concourse_from_gate(gate)
    gate_letter = _concourse_letter_from_label(gate_concourse)
    if gate_letter:
        ranked = []
        for it in candidates:
            it_letter = _concourse_letter_from_label(it.get("concourse", ""))
            score = 1 if it_letter == gate_letter else 0
            ranked.append((score, it))
        ranked.sort(key=lambda x: x[0], reverse=True)
        if ranked and ranked[0][0] > 0:
            return ranked[0][1]

    return candidates[0]

# -------- Random defaults --------
DEFAULT_GATES = [
    "C31","C24","D1","D3","B5","B10","A11","A9","E2","E4",
    "E7","A5","C15","D6","C20","B1","D9","E12","A3","C12"
]
def random_gate(): return random.choice(DEFAULT_GATES)
def random_hours_4_to_12(): return f"{random.randint(4,12)} hours"

# -------- Cuisine / intent helpers --------
CUISINES = [
    "indian","pakistani","lebanese","italian","french","japanese","thai","chinese",
    "american","mexican","turkish","korean","arabic","mediterranean","seafood","burger",
    "pizza","vegetarian","vegan","dessert","coffee"
]
YES_WORDS = {"yes","y","yeah","yep","sure","okay","ok","please","go ahead","do it"}
NO_WORDS  = {"no","n","nope","not now","later","don’t","dont"}

def extract_cuisine(text: str) -> str:
    t = (text or "").lower()
    for c in CUISINES:
        if re.search(rf"\b{re.escape(c)}\b", t):
            return c
    return ""

def says_yes(text: str) -> bool:
    t = (text or "").lower().strip()
    return any(w in t for w in YES_WORDS)

def says_no(text: str) -> bool:
    t = (text or "").lower().strip()
    return any(w in t for w in NO_WORDS)

def cuisine_exists_in_concourse(cuisine: str, concourse: str) -> bool:
    if not cuisine or not concourse:
        return False
    c = cuisine.lower()
    conc = concourse.lower()
    for item in embedded_chunks:
        t = item["text"].lower()
        if c in t and conc in t:
            return True
    return False

# -------- Itinerary helpers --------
def init_itinerary():
    if "itinerary" not in session:
        session["itinerary"] = {"Dining": [], "Shopping": [], "Relax": []}

def add_to_itinerary(category: str, place: str, description: str = "", concourse: str = "",
                     walk_time: str = "", image_url: str = "", item_id=None, location_ids=None):
    """
    Append a saved place to the user's itinerary (de-dupe by name+id).
    """
    init_itinerary()
    cat = category.title()
    if cat not in session["itinerary"]:
        session["itinerary"][cat] = []
    for it in session["itinerary"][cat]:
        if it.get("name","").lower() == (place or "").lower() and str(it.get("id")) == str(item_id):
            return
    session["itinerary"][cat].append({
        "id": item_id,
        "location_ids": location_ids or [],
        "name": place,
        "description": description,
        "concourse": concourse,
        "walk_time": walk_time,
        "image": image_url
    })
    session.modified = True

CATEGORY_WORDS = {
    "dining": {"dining","eat","food","restaurant","cuisine","breakfast","lunch","dinner","snack","meal","drink","coffee","bar","cafe"},
    "shopping": {"shopping","shop","buy","stores","boutique","retail","gift","souvenir","clothes","apparel","electronics","books","magazine"},
    "relax": {"relax","lounge","spa","rest","massage","quiet","meditate","yoga","nap","sleep","chill","unwind","calm"}
}
def detect_category_from_text(text: str) -> str:
    t = (text or "").lower()
    for cat, words in CATEGORY_WORDS.items():
        if any(w in t for w in words):
            return cat.title()
    return session.get("current_category", "Dining")

# Parse numbered recommendations (AI returns HTML; we parse last bubble)
RECO_REGEX = re.compile(
    r"(\d+)\.\s*<strong>(.*?)</strong>\s*—\s*(.*?)<br>\s*<span[^>]*>(.*?)</span>\s*<br>\s*<span[^>]*>(.*?)</span>",
    re.S | re.I
)
def parse_recos_from_html(html: str):
    return RECO_REGEX.findall(html or "")

# -------- Routes --------
@app.route("/", methods=["GET", "POST"])
def chat():
    """
    - GET (fresh): pre-scan welcome (disabled chat).
    - POST (from scanner): fresh session, save passenger, choose random gate/time if missing, show greeting on GET.
    - GET (after scan): inject first AI greeting + show_first_options buttons.
    """
    if "history" not in session:
        session["history"] = []

    pre_scan = False
    if not session.get("passenger_name") and request.method == "GET" and not session.get("history"):
        pre_scan = True

    if request.method == "POST":
        # Coming from scanner
        session.clear()
        session["history"] = []

        # We accept generic names/flight/etc. from the scanner page
        session["passenger_name"] = (request.form.get("passenger_name") or "").strip() or "Passenger"
        # For consistency in templates, keep the legacy key flight_number
        session["flight_number"]  = (request.form.get("flight_number")  or request.form.get("flight") or "").strip() or "Unknown"
        gate_in = (request.form.get("gate") or "").strip().upper()
        session["gate"] = gate_in or random_gate()
        session["time"] = (request.form.get("time") or "").strip()
        session["time_to_flight"] = random_hours_4_to_12()

        session["just_scanned"] = True
        session.modified = True
        return redirect(url_for("chat"))

    if session.pop("just_scanned", False):
        passenger_name  = (session.get("passenger_name") or "Passenger").upper()
        destination     = "London (LHR)"  # Could be mapped from flight if available
        time_to_flight  = session.get("time_to_flight") or random_hours_4_to_12()
        gate            = session.get("gate") or random_gate()
        flight          = session.get("flight_number") or "Unknown"

        first_msg = (
            f"Hi {passenger_name}, your next flight {flight} to {destination} departs in {time_to_flight} "
            f"from Gate {gate}. What would you like to do while you're here?"
        )
        session["history"].append({"role": "assistant", "content": first_msg})
        session["show_first_options"] = True
        session.modified = True

    show_first_options = session.pop("show_first_options", False)

    return render_template(
        "index.html",
        history=session.get("history", []),
        passenger_name=session.get("passenger_name", ""),
        flight_number=session.get("flight_number", ""),
        gate=session.get("gate", "") or random_gate(),
        time=session.get("time", ""),
        pending_question="",
        show_first_options=show_first_options,
        pre_scan=pre_scan
    )

@app.route("/ask", methods=["POST"])
def ask():
    """
    - Handles numeric selection or “add X to my <cat> itinerary” (with location_ids).
    - Respects cuisine near gate; if none, asks to expand to nearby concourses.
    - Otherwise, generates options with required HTML formatting.
    """
    if "history" not in session:
        session["history"] = []

    data = request.get_json(force=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"assistant": "No question provided."}), 400

    # Track category intent (used if the next user reply is a number)
    session["current_category"] = detect_category_from_text(question)
    session.modified = True

    passenger_name = (data.get("passenger_name") or session.get("passenger_name") or "Passenger").strip()
    flight_number  = (data.get("flight_number")  or session.get("flight_number")  or "Unknown").strip()
    gate           = (data.get("gate")          or session.get("gate")          or random_gate()).strip().upper()
    boarding_time  = (data.get("time")          or session.get("time")          or "").strip()

    session["history"].append({"role": "user", "content": question})

    # ----- Explicit add: "add X to my <cat> itinerary"
    m_add = re.search(r"add\s+(.+?)\s+(?:to|into)\s+my\s+(dining|shopping|relax)\s+itinerary", question, flags=re.I)
    if m_add:
        place = m_add.group(1).strip()
        cat   = m_add.group(2).title()

        matched = best_catalog_match(place, gate=gate)
        if matched:
            add_to_itinerary(
                cat,
                matched["name"],
                matched.get("description",""),
                matched.get("concourse",""),
                "within 5 min walk",
                matched.get("image"),
                item_id=matched.get("id"),
                location_ids=matched.get("location_ids") or []
            )
        else:
            # Fallback: parse last AI options
            last_ai_html = ""
            for msg in reversed(session["history"]):
                if msg["role"] == "assistant":
                    last_ai_html = msg["content"] or ""
                    break
            desc = walk = conc = ""
            for _, name, d, w, c in parse_recos_from_html(last_ai_html):
                if name.strip().lower() == place.lower():
                    desc, walk, conc = d, w, c
                    break
            add_to_itinerary(cat, place, desc, conc, walk)

        reply = f"{place} has been added to your {cat} itinerary. 🧾"
        session["history"].append({"role": "assistant", "content": reply})
        session.modified = True
        return jsonify({"assistant": reply})

    # ----- Numeric selection (e.g. "2")
    if re.fullmatch(r"\d+", question):
        choice_num = int(question)
        last_ai_html = ""
        for msg in reversed(session["history"]):
            if msg["role"] == "assistant":
                last_ai_html = msg["content"] or ""
                break
        matches_html = parse_recos_from_html(last_ai_html)
        if matches_html and 1 <= choice_num <= len(matches_html):
            _, place, desc, walk, conc = matches_html[choice_num - 1]
            match_obj = best_catalog_match(place, gate=gate)
            if match_obj:
                add_to_itinerary(
                    session.get("current_category", "Dining"),
                    match_obj["name"],
                    match_obj.get("description", desc),
                    match_obj.get("concourse", conc),
                    walk or "within 5 min walk",
                    match_obj.get("image"),
                    item_id=match_obj.get("id"),
                    location_ids=match_obj.get("location_ids") or []
                )
            else:
                add_to_itinerary(
                    session.get("current_category", "Dining"),
                    place, desc, conc, walk
                )

            reply = f"{place} has been added to your {session.get('current_category','Dining')} itinerary. 🧾"
            session["history"].append({"role": "assistant", "content": reply})
            session.modified = True
            return jsonify({"assistant": reply})

        nudger = "I couldn’t interpret that selection. Pick a number from the latest list, or say “add <name> to my itinerary”."
        session["history"].append({"role": "assistant", "content": nudger})
        session.modified = True
        return jsonify({"assistant": nudger})

    # ----- Cuisine-aware recommendations
    time_to_flight = session.get("time_to_flight") or random_hours_4_to_12()
    destination    = "London (LHR)"
    concourse_name = infer_concourse_from_gate(gate)
    cuisine = extract_cuisine(question)

    # Previously asked to expand?
    if session.get("awaiting_cuisine_expand"):
        if says_yes(question):
            cuisine = session.pop("pending_cuisine", "")
            session.pop("awaiting_cuisine_expand", None)
            context = search_similar(client, EMBED_MODEL, question, embedded_chunks, gate="", cuisine=cuisine)
            system_prompt = (
                "List 3–5 options for the requested CUISINE anywhere in the airport. "
                "FORMAT (HTML):\n"
                "1. <strong>Place Name</strong> — short description<br>\n"
                "<span style='color:#8a8a8a;font-style:italic;'>located within 5 min walk</span><br>\n"
                "<span style='color:#8a8a8a;'>Concourse A, Level 1</span>\n\n"
                "End with: 'Reply with the number to save it to your itinerary.'"
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content":
                    f"(Known context)\nPassenger name: {passenger_name}\nFlight number: {flight_number}\n"
                    f"Destination: {destination}\nTime until boarding: {time_to_flight}\nBoarding gate: {gate}\n"
                    f"Cuisine: {cuisine or 'none'}\n\n(Reference data)\n{context}"
                }
            ] + session["history"]
            response = client.chat_completions.create(  # back-compat alias
                model=CHAT_MODEL, messages=messages, temperature=0.7, max_tokens=700
            ) if hasattr(client, "chat_completions") else client.chat.completions.create(
                model=CHAT_MODEL, messages=messages, temperature=0.7, max_tokens=700
            )
            assistant_reply = response.choices[0].message.content
            session["history"].append({"role": "assistant", "content": assistant_reply})
            session.modified = True
            return jsonify({"assistant": assistant_reply})
        elif says_no(question):
            session.pop("pending_cuisine", None)
            session.pop("awaiting_cuisine_expand", None)
            assistant_reply = "No problem. Is there another cuisine you’d like to try instead (e.g., Lebanese, Italian, Thai)?"
            session["history"].append({"role": "assistant", "content": assistant_reply})
            session.modified = True
            return jsonify({"assistant": assistant_reply})
        else:
            assistant_reply = "Just to confirm — should I search nearby concourses for more options? (Yes/No)"
            session["history"].append({"role": "assistant", "content": assistant_reply})
            session.modified = True
            return jsonify({"assistant": assistant_reply})

    # If cuisine requested but not in current concourse → ask to expand
    if cuisine:
        if not cuisine_exists_in_concourse(cuisine, concourse_name):
            session["pending_cuisine"] = cuisine
            session["awaiting_cuisine_expand"] = True
            nearby_hint = "nearby concourses (A, C, D, E)" if concourse_name in {
                "Concourse B","Concourse A","Concourse C","Concourse D","Concourse E"
            } else "other concourses"
            assistant_reply = (
                f"I couldn’t find any {cuisine.title()} options near {concourse_name}. "
                f"Would you like me to search {nearby_hint} for more choices?"
            )
            session["history"].append({"role": "assistant", "content": assistant_reply})
            session.modified = True
            return jsonify({"assistant": assistant_reply})

    # Normal local search
    context = search_similar(client, EMBED_MODEL, question, embedded_chunks, gate=gate, cuisine=cuisine)
    system_prompt = (
        "You are a helpful airport assistant. "
        "If the user picks Dining / Shopping / Relax, ask a brief clarifying question, then list 3–5 options.\n\n"
        "FORMAT EACH RECOMMENDATION EXACTLY (HTML):\n"
        "1. <strong>Place Name</strong> — short description<br>\n"
        "<span style='color:#8a8a8a;font-style:italic;'>located within 5 min walk</span><br>\n"
        "<span style='color:#8a8a8a;'>Concourse A, Level 1</span>\n\n"
        "After listing options, say: 'Reply with the number to save it to your itinerary.'\n"
        "If the requested CUISINE is not found near the gate, explicitly say so and ask permission to search nearby concourses."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content":
            f"(Known context)\nPassenger name: {passenger_name}\nFlight number: {flight_number}\n"
            f"Destination: {destination}\nTime until boarding: {time_to_flight}\nBoarding gate: {gate}\n"
            f"Cuisine (if any): {cuisine or 'none'}\n\n(Reference data)\n{context}"
        }
    ] + session["history"]

    # Azure SDK has both client.chat.completions and client.chat_completions in different versions
    response = client.chat_completions.create(  # back-compat alias if present
        model=CHAT_MODEL, messages=messages, temperature=0.7, max_tokens=700
    ) if hasattr(client, "chat_completions") else client.chat.completions.create(
        model=CHAT_MODEL, messages=messages, temperature=0.7, max_tokens=700
    )
    assistant_reply = response.choices[0].message.content

    # Preserve HTML (frontend should render with |safe or by inserting HTML)
    session["history"].append({"role": "assistant", "content": assistant_reply})
    session.modified = True
    return jsonify({"assistant": assistant_reply})

@app.route("/scanner", methods=["GET"])
def scanner():
    return render_template("scanner.html")

@app.route("/itinerary", methods=["GET"])
def itinerary():
    init_itinerary()
    # You can pass flight/gate/time here if your template uses them
    return render_template("itinerary.html", itinerary=session["itinerary"])

@app.route("/delete_itinerary/<category>/<int:index>", methods=["POST"])
def delete_itinerary(category, index):
    init_itinerary()
    if category in session["itinerary"] and 0 <= index < len(session["itinerary"][category]):
        del session["itinerary"][category][index]
        session.modified = True
    return redirect(url_for("itinerary"))

@app.route("/reset", methods=["GET"])
def reset():
    session.clear()
    return redirect(url_for("chat"))

if __name__ == "__main__":
    app.run(debug=True)
