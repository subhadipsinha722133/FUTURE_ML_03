# FUTURE_ML_03
Customer Support Chatbot - Future Interns ML Task 3
# 🤖 FUTURE_ML_03: Customer Support Chatbot

## Overview
A simple Flask-based customer support chatbot that uses machine learning to classify user queries into intents and respond accordingly.

## Files
├── chatbot.py           # Flask app main file<br>
├── model/               # Contains trained model and vectorizer files<br>
│   ├── intent_classifier.pkl<br>
│   └── vectorizer.pkl  <br>
├── scripts/             # Helper scripts, e.g., text preprocessing<br>
│   └── preprocess.py<br>
├── static/              # Static files like CSS<br>
│   └── style.css<br>
├── templates/           # HTML templates<br>
│   └── index.html<br>
├── data/                # Dataset used for training (optional)<br>
│   └── customer_support_tickets.csv<br>
├── requirements.txt     # Python dependencies<br>
└── README.md            # Project documentation<br>

## Features
- Intent classification of customer queries using a trained ML model.
- Handles common customer support intents like refund requests, billing inquiries, cancellations, technical issues, and product inquiries.
- Includes a fallback response for unknown queries.
- Simple web interface for chat interaction.
- Preprocessing and cleaning of user input text.
- Basic confidence thresholding to detect uncertain predictions.
- Hardcoded greeting detection for friendly user experience.


## Tools Used
- Python
- NLTK
- scikit-learn
- NumPy


## How to Run
```bash
pip install -r requirements.txt
python chatbot.py

