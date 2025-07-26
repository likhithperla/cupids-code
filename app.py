import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from google.cloud import firestore
from google.oauth2 import service_account
from google.api_core.exceptions import GoogleAPICallError

# --- Initialization ---
app = Flask(__name__)
CORS(app)

# --- Configuration ---
# The app will get its "passwords" from Render's Environment Variables
API_KEY = os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    print("Warning: GOOGLE_API_KEY environment variable not set.")

genai.configure(api_key=API_KEY)

# Initialize Firestore DB client
db = None
try:
    # This block tries to find the special database password from the environment
    gcp_json_credentials_str = os.environ.get("FIRESTORE_CREDENTIALS_JSON")
    if gcp_json_credentials_str:
        gcp_json_credentials_dict = json.loads(gcp_json_credentials_str)
        credentials = service_account.Credentials.from_service_account_info(gcp_json_credentials_dict)
        db = firestore.Client(credentials=credentials)
        print("Successfully initialized Firestore with provided credentials.")
    else:
        print("Warning: FIRESTORE_CREDENTIALS_JSON not found. Trying default credentials.")
        db = firestore.Client()

except Exception as e:
    print(f"CRITICAL: Could not initialize Firestore. Chat history will not be saved. Error: {e}")
    db = None

# Initialize AI Models
model = genai.GenerativeModel('gemini-1.5-flash')

# --- API Endpoints (No change in logic) ---

@app.route('/generate_opener', methods=['POST'])
def generate_opener():
    if not request.json or 'image_data' not in request.json or 'tone' not in request.json:
        return jsonify({"error": "Missing image data or tone"}), 400
    
    image_data = request.json['image_data']
    tone = request.json['tone']
    image_parts = [{"mime_type": "image/jpeg", "data": image_data}]
    
    prompt = f"""
    You are a charismatic and clever dating assistant. Your goal is to help a user write an excellent opening message based on a screenshot of their match's profile.
    Analyze the provided screenshot, paying attention to text and visuals.
    Based on your analysis, generate 3-5 distinct opening messages that are engaging, tailored, and reflect a "{tone}" tone.
    Provide your response ONLY as a valid JSON array of strings, like this:
    ["Is that a golden retriever?", "That hiking spot looks amazing! Was that taken around here?"]
    """
    
    try:
        response = model.generate_content([prompt, *image_parts])
        openers_json = json.loads(response.text.strip())
        return jsonify({"openers": openers_json})
    except (GoogleAPICallError, json.JSONDecodeError, Exception) as e:
        print(f"Error during opener generation: {e}")
        return jsonify({"error": "Failed to generate openers with AI."}), 500

@app.route('/chat', methods=['POST'])
def chat():
    if not request.json or 'history' not in request.json:
        return jsonify({"error": "Missing chat history"}), 400

    user_id = "user_12345"
    history = request.json['history']
    
    system_instruction = {"role": "user", "parts": [{"text": "You are Cupid's Code, a friendly, wise, and encouraging AI dating coach."}]}
    last_user_message = history[-1]['parts'][0]['text']

    try:
        chat_session = model.start_chat(history=[system_instruction, *history])
        response = chat_session.send_message(last_user_message)
        ai_reply = response.text

        if db:
            try:
                chat_ref = db.collection('users').document(user_id).collection('chat_history')
                chat_ref.add({'role': 'user', 'text': last_user_message, 'timestamp': firestore.SERVER_TIMESTAMP})
                chat_ref.add({'role': 'model', 'text': ai_reply, 'timestamp': firestore.SERVER_TIMESTAMP})
            except Exception as e:
                print(f"Error saving chat history to Firestore: {e}")
        
        return jsonify({"reply": ai_reply})
    except (GoogleAPICallError, Exception) as e:
        print(f"Error during chat: {e}")
        return jsonify({"error": "Failed to get chat response from AI."}), 500

# --- Main Execution ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
