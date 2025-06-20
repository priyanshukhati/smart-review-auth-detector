import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Create model directory if it doesn't exist
os.makedirs("model", exist_ok=True)

# Load dataset
df = pd.read_csv("dataset/fake_review_dataset.csv")

# Inspect raw labels
print("Raw label values:", df['label'].unique())
print("Label counts (before filtering):")
print(df['label'].value_counts())
print("Original columns:", df.columns)

# Clean and select required data
df = df[['text_', 'label']].dropna()

# Convert labels to uppercase and map: CG = 1 (Genuine), OR = 0 (Fake)
df['label'] = df['label'].str.upper()
df = df[df['label'].isin(['CG', 'OR'])]
df['label'] = df['label'].map({'CG': 1, 'OR': 0})

# Check final label distribution
print("Final label distribution:")
print(df['label'].value_counts())

# Ensure at least two classes exist
if len(df['label'].unique()) < 2:
    raise ValueError("Need at least 2 classes in the dataset to train the model.")

# Prepare features and labels
X = df['text_']
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training set label distribution:")
print(y_train.value_counts())

# TF-IDF vectorization
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Evaluate model
y_pred = model.predict(X_test_tfidf)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model and vectorizer
with open("model/review_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

print("✅ Model and vectorizer saved successfully!")
