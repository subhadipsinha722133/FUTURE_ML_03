# scripts/train_model.py

import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from preprocess import clean_text

# Load preprocessed data
df = pd.read_csv('data/customer_support_tickets.csv')  # or saved cleaned file
df = df[['Ticket Description', 'Ticket Type']].dropna()
df.rename(columns={'Ticket Description': 'text', 'Ticket Type': 'intent'}, inplace=True)

# Clean text
df['cleaned_text'] = df['text'].apply(clean_text)

# Features and labels
X = df['cleaned_text']
y = df['intent']

# Print class distribution to check imbalance
print("Intent distribution:\n", y.value_counts())

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize text data with unigrams + bigrams
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model with class weights and higher max iterations
model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Show some misclassified examples
print("\nSome misclassified samples:")
misclassified_mask = y_test != y_pred
misclassified_texts = X_test[misclassified_mask]
misclassified_true = y_test[misclassified_mask]
misclassified_pred = [pred for pred, mask in zip(y_pred, misclassified_mask) if mask]

for i, text in enumerate(misclassified_texts.head(5)):
    print(f"\nText: {text}")
    print(f"True: {misclassified_true.iloc[i]}")
    print(f"Predicted: {misclassified_pred[i]}")

# Save the model and vectorizer
joblib.dump(model, 'model/intent_classifier.pkl')
joblib.dump(vectorizer, 'model/vectorizer.pkl')
