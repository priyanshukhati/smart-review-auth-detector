import pickle

# Load your old model and vectorizer (these should already be present in your project)
with open("model/review_model.pkl", "rb") as f:
    review_model = pickle.load(f)

with open("model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Re-save them using current Python & numpy version (which matches Streamlit cloud)
with open("model/review_model.pkl", "wb") as f:
    pickle.dump(review_model, f)

with open("model/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Model and vectorizer re-saved successfully.")
