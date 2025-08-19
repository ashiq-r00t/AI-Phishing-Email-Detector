# Day 3: Dataset Exploration and Model Training

import pandas as pd
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Set dataset path - adjust username accordingly
dataset_path = '/Users/your-username/Desktop/email dataset/'
file_name = 'phishing_email.csv'
file_path = os.path.join(dataset_path, file_name)

# Load dataset
df = pd.read_csv(file_path)

# Display first few rows and info
print(df.head())
print(df.info())

# Label distribution
if 'label' in df.columns:
    print(df['label'].value_counts())
else:
    print("No 'label' column in dataset")

# Display sample email texts
for i in range(3):
    print(f"Email {i+1} text:\n{df['text_combined'].iloc[i]}\n")

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Apply cleaning
df['cleaned_text'] = df['text_combined'].apply(clean_text)

# Show original vs cleaned for first 5 emails
print(df[['text_combined', 'cleaned_text']].head())

# Vectorize text with TF-IDF
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X = tfidf.fit_transform(df['cleaned_text'])
print(f"TF-IDF matrix shape: {X.shape}")

# Prepare labels
y = df['label']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape} samples")

# Train logistic regression classifier
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
