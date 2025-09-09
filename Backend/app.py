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

# ✅ Explicit CORS setup (fix Render error)
frontend_origins = [
    "http://localhost:5173",   # local dev
    "https://mindful-matrix-5.onrender.com"  # deployed frontend
]
CORS(app, resources={r"/*": {"origins": frontend_origins}}, supports_credentials=True)

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
# Routes (signup, login, chat, moods etc.)
# ==========================
# 🔹 Keeping all your existing routes unchanged...

@app.route("/test", methods=["GET"])
def test():
    return jsonify({"status": "ok", "message": "Flask server running!"})

# ==========================
# MAIN
# ==========================
if __name__ == "__main__":
    app.run(debug=True)
