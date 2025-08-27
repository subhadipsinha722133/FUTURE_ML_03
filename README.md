# FUTURE_ML_03
Customer Support Chatbot - Future Interns ML Task 3
# 🤖 FUTURE_ML_03: Customer Support Chatbot

## Overview
A simple Flask-based customer support chatbot that uses machine learning to classify user queries into intents and respond accordingly.

## Files
├── chatbot.py           # Flask app main file
├── model/               # Contains trained model and vectorizer files
│   ├── intent_classifier.pkl
│   └── vectorizer.pkl
├── scripts/             # Helper scripts, e.g., text preprocessing
│   └── preprocess.py
├── static/              # Static files like CSS
│   └── style.css
├── templates/           # HTML templates
│   └── index.html
├── data/                # Dataset used for training (optional)
│   └── customer_support_tickets.csv
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation

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

