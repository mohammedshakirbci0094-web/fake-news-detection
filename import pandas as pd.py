import pandas as pd
import numpy as np
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load datasets
true_df = pd.read_csv("True.csv")
fake_df = pd.read_csv("Fake.csv")

# Add labels
true_df["label"] = 1   # 1 = True news
fake_df["label"] = 0   # 0 = Fake news

# Combine datasets
df = pd.concat([true_df, fake_df], axis=0).reset_index(drop=True)

# Use only the text column (title + text for better results)
df["content"] = df["title"].astype(str) + " " + df["text"].astype(str)

# Basic text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)  # remove URLs
    text = re.sub(r"[^a-zA-Z]", " ", text)  # keep only letters
    text = re.sub(r"\s+", " ", text)  # remove extra spaces
    return text

df["content"] = df["content"].apply(clean_text)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df["content"], df["label"], test_size=0.2, random_state=42
)

# Convert text to numerical features using TF-IDF
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Predictions
y_pred = model.predict(X_test_tfidf)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
