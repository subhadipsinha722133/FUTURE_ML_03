import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, request, jsonify, render_template
import joblib
from scripts.preprocess import clean_text


# Load model and vectorizer
model = joblib.load('model/intent_classifier.pkl')
vectorizer = joblib.load('model/vectorizer.pkl')

# Intent-response mapping (for now, hardcoded)
response_map = {
    'Greeting': "Hello! How can I assist you today?",   
    'Billing inquiry': "Sure! Let me help you with billing.",
    'Cancellation request': "I can help you cancel your service.",
    'Product inquiry': "Here's more info about the product.",
    'Refund request': "I'll assist you with the refund process.",
    'Technical issue': "Let me troubleshoot that issue for you."
}

# Create Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


# chatbot.py (modify /chat route)
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_msg = data.get('message', '')
    
    if not user_msg:
        return jsonify({'error': 'Empty message'}), 400
    
    cleaned = clean_text(user_msg)
    print("Cleaned message:", cleaned)  # Debug print
    
    vectorized = vectorizer.transform([cleaned])
    predicted_intent = model.predict(vectorized)[0]
    
    print("Predicted intent:", predicted_intent)  # Debug print
    
    response = response_map.get(predicted_intent, "I'm not sure how to help with that.")
    
    print("Response:", response)  # Debug print

    return jsonify({
        'intent': predicted_intent,
        'response': response
    })

if __name__ == '__main__':
    app.run(debug=True)
