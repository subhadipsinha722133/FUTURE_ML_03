
import json
import random
import nltk
import numpy as np

from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('punkt')

stemmer = PorterStemmer()

# Load intents
with open("intents.json") as file:
    data = json.load(file)

# Prepare patterns and responses
corpus = []
tags = []
responses = {}

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        corpus.append(pattern)
        tags.append(intent["tag"])
    responses[intent["tag"]] = intent["responses"]

# Preprocess
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(words):
    return [stemmer.stem(w.lower()) for w in words]

stemmed_corpus = [" ".join(stem(tokenize(pattern))) for pattern in corpus]

# Vectorize
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(stemmed_corpus).toarray()

def predict_class(user_input):
    input_tokens = stem(tokenize(user_input))
    input_str = " ".join(input_tokens)
    input_vec = vectorizer.transform([input_str]).toarray()

    similarities = np.dot(X, input_vec.T).flatten()
    best_match_index = np.argmax(similarities)
    return tags[best_match_index]

def get_response(tag):
    return random.choice(responses[tag])

# Run chatbot
print("🤖 Chatbot is running! Type 'quit' to exit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Bot: Goodbye!")
        break
    tag = predict_class(user_input)
    response = get_response(tag)
    print(f"Bot: {response}")
