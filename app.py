import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from google.cloud import firestore
from google.api_core.exceptions import GoogleAPICallError

# --- Initialization ---
app = Flask(__name__)
CORS(app)

# --- Configuration ---
API_KEY = os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    print("Warning: GOOGLE_API_KEY environment variable not set. This is required for local testing.")

genai.configure(api_key=API_KEY)

try:
    db = firestore.Client()
except Exception as e:
    print(f"Warning: Could not initialize Firestore. Chat history will not be saved. Error: {e}")
    db = None

# Use a single, powerful model for all tasks
model = genai.GenerativeModel('gemini-1.5-flash')

# --- API Endpoints ---

@app.route('/generate_opener', methods=['POST'])
def generate_opener():
    """
    Analyzes a match's profile screenshot and generates opening lines.
    """
    if not request.json or 'image_data' not in request.json or 'tone' not in request.json:
        return jsonify({"error": "Missing image data or tone"}), 400
    
    image_data = request.json['image_data']
    tone = request.json['tone']
    image_parts = [{"mime_type": "image/jpeg", "data": image_data}]
    
    prompt = f"""
    You are a charismatic and clever dating assistant. Your goal is to help a user write an excellent opening message based on a screenshot of their match's profile.

    Analyze the provided screenshot, paying attention to:
    - Text in the bio: Look for hobbies, interests, questions, or anything unique.
    - Visuals in the photo(s): Look for locations, activities, pets, clothing, or general vibe.

    Based on your analysis, generate 3-5 distinct opening messages that are:
    - Engaging and likely to get a response.
    - Tailored to the person's profile.
    - Reflective of a "{tone}" tone.
    - Formatted as a question or a playful observation. Avoid generic compliments like "you're cute".

    Provide your response ONLY as a valid JSON array of strings, like this:
    ["Is that a golden retriever? My family's dog looks just like him!", "That hiking spot looks amazing! Was that taken around here?", "The bio says you're a coffee snob... what's the best cafe in town in your expert opinion?"]
    """
    
    try:
        response = model.generate_content([prompt, *image_parts])
        # The response should be a clean JSON array string
        openers_json = json.loads(response.text.strip())
        return jsonify({"openers": openers_json})
    except (GoogleAPICallError, json.JSONDecodeError, Exception) as e:
        print(f"Error during opener generation: {e}")
        return jsonify({"error": "Failed to generate openers with AI. The profile might be hard to read."}), 500

@app.route('/chat', methods=['POST'])
def chat():
    """
    Handles the dating coach chat, saving history to Firestore.
    """
    if not request.json or 'history' not in request.json:
        return jsonify({"error": "Missing chat history"}), 400

    user_id = "user_12345"  # In a real app, this would come from an authentication system
    history = request.json['history']
    
    system_instruction = {"role": "user", "parts": [{"text": "You are Cupid's Code, a friendly, wise, and encouraging AI dating coach. Provide helpful, safe, and positive advice. Keep responses supportive and concise."}]}
    
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
    app.run(host='127.0.0.1', port=8080, debug=True)
