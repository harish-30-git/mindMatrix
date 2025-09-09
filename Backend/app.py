from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import google.generativeai as genai
from dotenv import load_dotenv
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from scipy.sparse import hstack, csr_matrix
import bcrypt
import jwt
from datetime import datetime, timedelta, timezone
from bson.objectid import ObjectId
from functools import wraps
from pymongo import MongoClient, ASCENDING, ReturnDocument

# ==========================
# ENV + FLASK CONFIG
# ==========================
load_dotenv()
app = Flask(__name__)

# ✅ Allow only specific frontend domains
allowed_origins = [
    "https://mindful-matrix-lliz.vercel.app",  # Your deployed frontend on Vercel
    "http://localhost:5173"  # Local dev (Vite default)
]
CORS(app, resources={r"/*": {"origins": allowed_origins}}, supports_credentials=True)

app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "supersecretkey")
mongo_uri = os.getenv("MONGO_URI")
client = MongoClient(mongo_uri)
db = client.get_database()
users = db.users
moods = db.moods
moods.create_index([("user_id", ASCENDING), ("date", ASCENDING)], unique=True)

# ==========================
# GEMINI CONFIG
# ==========================
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

system_prompt = (
    "You are MentalHealthBot, a supportive assistant designed to help with "
    "mental health awareness, emotional well-being, stress management, and "
    "healthy lifestyle habits. You will respond in a compassionate, encouraging, "
    "and non-judgmental way. "
    "Keep your responses brief and concise, preferably 2–3 sentences. "
    "You remember the user's previous messages in the current conversation, "
    "so your advice is context-aware and continuous. "
    "Respond naturally and conversationally, as a human would, without repeating "
    "greetings like 'Hi there' after the first interaction. "
    "Provide actionable tips, reassurance, or short reflective questions to help the user. "
    "If the user asks something unrelated to mental health, reply: "
    "\"Sorry, I don’t have access to other resources, I’m a Mental Health Bot.\""
)

# ==========================
# ML MODEL SETUP
# ==========================
print("🔹 Loading dataset and training ML model...")
text_data = pd.read_csv("fully_cleaned_dataset.csv")
text_data = text_data.drop(columns=[col for col in text_data.columns if col.startswith("Unnamed")], errors="ignore")

categorical_cols = [
    'Gender', 'Country', 'Occupation', 'self_employed', 'family_history',
    'treatment', 'Days_Indoors', 'Growing_Stress', 'Changes_Habits',
    'Mental_Health_History', 'Mood_Swings', 'Coping_Struggles',
    'Work_Interest', 'Social_Weakness', 'mental_health_interview',
    'care_options'
]
categorical_cols = [col for col in categorical_cols if col in text_data.columns]

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    text_data[col] = le.fit_transform(text_data[col].astype(str))
    label_encoders[col] = le

text_data['statement'] = text_data['statement'].fillna("").astype(str)
text_data = text_data[text_data['statement'].str.strip() != ""]
vectorizer = TfidfVectorizer(stop_words='english')
text_features = vectorizer.fit_transform(text_data['statement'])

X_survey = text_data.drop(columns=['Timestamp', 'statement', 'status'], errors='ignore')
X = hstack([csr_matrix(X_survey.values), text_features])
y = text_data['status']
survey_columns = X_survey.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

print("\n=== Model Evaluation ===")
print(classification_report(y_test, model.predict(X_test)))
print("✅ Model ready for predictions!")

# ==========================
# Helper functions
# ==========================
def get_contextual_reply(current_user, user_text, prediction_status="N/A", input_type="user"):
    history = current_user.get("profile", {}).get("chatHistory", [])
    conversation_history = []
    for h in history[-10:]:
        if "user" in h and "bot" in h:
            conversation_history.append({"role": "user", "content": h["user"]})
            conversation_history.append({"role": "assistant", "content": h["bot"]})
        elif "user_input" in h and "bot_response" in h:
            conversation_history.append({"role": "user", "content": h["user_input"]})
            conversation_history.append({"role": "assistant", "content": h["bot_response"]})

    full_prompt = system_prompt + "\n\nConversation so far:\n"
    for msg in conversation_history:
        role = "User" if msg["role"] == "user" else "Bot"
        full_prompt += f"{role}: {msg['content']}\n"
    full_prompt += f"User: [Predicted Status: {prediction_status}] {user_text}\nBot:"

    try:
        response = gemini_model.generate_content(full_prompt)
        return response.text.strip() if response.text else "Sorry, I couldn't generate a response."
    except Exception as e:
        print("⚠ Gemini API error:", e)
        return "Sorry, I couldn't generate a response at this time."

def map_mood_to_survey(mood):
    mood_map = {"happy": 2, "neutral": 1, "stressed": 0, "anxious": 0}
    survey_inputs = [0] * len(survey_columns)
    if "Mood_Swings" in survey_columns and mood in mood_map:
        idx = survey_columns.index("Mood_Swings")
        survey_inputs[idx] = mood_map[mood]
    return survey_inputs

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if 'Authorization' in request.headers:
            try:
                token = request.headers['Authorization'].split()[1]
            except IndexError:
                pass
        if not token:
            return jsonify({"msg": "Token is missing!"}), 401
        try:
            data = jwt.decode(token, app.config["SECRET_KEY"], algorithms=["HS256"])
            current_user = users.find_one({"_id": ObjectId(data["user_id"])}, {"passwordHash": 0})
            if not current_user:
                return jsonify({"msg": "User not found"}), 404
        except Exception as e:
            return jsonify({"msg": "Invalid token", "error": str(e)}), 401
        return f(current_user, *args, **kwargs)
    return decorated

def today_utc_iso():
    return datetime.now(timezone.utc).date().isoformat()

def iso_to_weekday_label(iso_date: str):
    dt = datetime.fromisoformat(iso_date).date()
    return dt.strftime("%a")

def last_7_dates_iso():
    today = datetime.now(timezone.utc).date()
    return [(today - timedelta(days=i)).isoformat() for i in range(6, -1, -1)]

# ==========================
# Authentication Routes
# ==========================
@app.route("/signup", methods=["POST"])
def signup():
    data = request.json
    username = data.get("username")
    email = data.get("email")
    password = data.get("password")
    if not all([username, email, password]):
        return jsonify({"msg": "Missing fields"}), 400
    if users.find_one({"email": email}):
        return jsonify({"msg": "Email already registered"}), 400

    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
    user = {
        "username": username,
        "email": email,
        "passwordHash": hashed,
        "profile": {
            "chatHistory": [],
            "preferences": {},
            "createdAt": datetime.now(timezone.utc)
        }
    }
    inserted = users.insert_one(user)
    token = jwt.encode(
        {"user_id": str(inserted.inserted_id), "exp": datetime.now(timezone.utc) + timedelta(hours=24)},
        app.config["SECRET_KEY"], algorithm="HS256"
    )
    return jsonify({"msg": "User registered successfully", "token": token, "username": username}), 201

@app.route("/login", methods=["POST"])
def login():
    data = request.json
    email = data.get("email")
    password = data.get("password")
    user = users.find_one({"email": email})
    if not user or not bcrypt.checkpw(password.encode("utf-8"), user["passwordHash"]):
        return jsonify({"msg": "Invalid credentials"}), 401
    token = jwt.encode(
        {"user_id": str(user["_id"]), "exp": datetime.now(timezone.utc) + timedelta(hours=24)},
        app.config["SECRET_KEY"], algorithm="HS256"
    )
    return jsonify({"token": token, "username": user["username"]})

@app.route("/profile", methods=["GET"])
@token_required
def profile(current_user):
    return jsonify({
        "username": current_user["username"],
        "email": current_user["email"],
        "profile": current_user.get("profile", {})
    })

# ==========================
# Interactive Chat Route
# ==========================
@app.route("/chat", methods=["POST"])
@token_required
def chat(current_user):
    user_message = request.json.get("message", "")
    reply = get_contextual_reply(current_user, user_message)
    users.update_one(
        {"_id": ObjectId(current_user["_id"])},
        {"$push": {"profile.chatHistory": {"user": user_message, "bot": reply, "timestamp": datetime.now(timezone.utc)}}}
    )
    return jsonify({"response": reply})

# ==========================
# Reset Chat / New Session Route
# ==========================
@app.route("/chat/reset", methods=["POST"])
@token_required
def reset_chat(current_user):
    users.update_one(
        {"_id": ObjectId(current_user["_id"])},
        {"$set": {"profile.chatHistory": []}}
    )
    return jsonify({"msg": "Conversation reset successfully"})

# ==========================
# Suggestions Route
# ==========================
@app.route("/api/get-suggestions", methods=["POST"])
@token_required
def get_suggestions(current_user):
    data = request.json
    mood = data.get("mood")
    journal_entry = data.get("journal_entry", "")

    if not mood and not journal_entry:
        return jsonify({"error": "Please provide mood or journal entry"}), 400

    survey_inputs = map_mood_to_survey(mood)
    text_feat = vectorizer.transform([journal_entry if journal_entry else mood])
    combined_features = hstack([csr_matrix([survey_inputs]), text_feat])

    expected_features = model.n_features_in_
    current_features = combined_features.shape[1]
    if current_features < expected_features:
        padding = csr_matrix((1, expected_features - current_features))
        combined_features = hstack([combined_features, padding])
    elif current_features > expected_features:
        combined_features = combined_features[:, :expected_features]

    predicted_status = model.predict(combined_features)[0]
    gemini_reply = get_contextual_reply(current_user, journal_entry if journal_entry else mood, predicted_status, input_type="journal")

    simple_mood = "others"
    if mood:
        simple_mood = mood.lower() if mood.lower() in ["happy", "neutral", "stressed", "anxious"] else "others"
    else:
        text_lower = journal_entry.lower()
        if "happy" in text_lower: simple_mood = "happy"
        elif "neutral" in text_lower: simple_mood = "neutral"
        elif "stressed" in text_lower: simple_mood = "stressed"
        elif "anxious" in text_lower: simple_mood = "anxious"

    users.update_one(
        {"_id": ObjectId(current_user["_id"])},
        {"$push": {"profile.chatHistory": {"user_input": journal_entry, "bot_response": gemini_reply, "timestamp": datetime.now(timezone.utc)}}}
    )

    return jsonify({
        "mood": simple_mood,
        "status": predicted_status,
        "suggestion": gemini_reply
    })

# ==========================
# Mood Tracking Routes
# ==========================
@app.route("/api/save-mood", methods=["POST"])
@token_required
def save_mood(current_user):
    body = request.get_json(force=True) or {}
    mood = (body.get("mood") or "others").lower()
    score = body.get("score")
    date_iso = body.get("date") or today_utc_iso()

    if score is None:
        return jsonify({"error": "score is required (0-10)"}), 400
    try:
        score = int(score)
    except Exception:
        return jsonify({"error": "score must be an integer 0-10"}), 400
    if not (0 <= score <= 10):
        return jsonify({"error": "score must be between 0 and 10"}), 400

    doc = {
        "user_id": str(current_user["_id"]),
        "date": date_iso,
        "weekday": iso_to_weekday_label(date_iso),
        "mood": mood,
        "score": score,
        "updatedAt": datetime.now(timezone.utc)
    }

    updated = moods.find_one_and_update(
        {"user_id": doc["user_id"], "date": doc["date"]},
        {"$set": doc, "$setOnInsert": {"createdAt": datetime.now(timezone.utc)}},
        upsert=True,
        return_document=ReturnDocument.AFTER
    )
    return jsonify({"ok": True, "saved": {
        "date": updated["date"],
        "weekday": updated["weekday"],
        "mood": updated["mood"],
        "score": updated["score"]
    }})

@app.route("/api/weekly-moods", methods=["GET"])
@token_required
def weekly_moods(current_user):
    user_id = str(current_user["_id"])
    last7 = last_7_dates_iso()
    docs = list(moods.find({"user_id": user_id, "date": {"$in": last7}}))
    by_date = {d["date"]: d for d in docs}
    result = []
    for iso in last7:
        d = by_date.get(iso)
        label = iso_to_weekday_label(iso)
        result.append({
            "day": label,
            "mood": int(d["score"]) if d and ("score" in d) else 0
        })
    return jsonify(result)

@app.route("/test", methods=["GET"])
def test():
    return jsonify({"status": "ok", "message": "Flask server running!"})

# ==========================
# MAIN
# ==========================
if __name__ == "__main__":
    app.run(debug=True)
