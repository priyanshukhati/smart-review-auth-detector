import streamlit as st
import pickle
import re

# ──────────────── PAGE CONFIG ──────────────── #
st.set_page_config(page_title="Fake Review Detector", page_icon="🕵️‍♂️", layout="centered")

# ──────────────── LOAD MODEL ──────────────── #
with open("model/review_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# ──────────────── SIDEBAR ──────────────── #
with st.sidebar:
    st.header("🛠 About this App")
    st.markdown("""
    **Fake Review Detector** helps identify whether a product review is likely **genuine** or **fake** using a trained ML model.

    **Inputs:**
    - Product name
    - Brand
    - Review platform
    - The actual review text

    **Output:**
    - Prediction (Genuine/Fake)
    - Model confidence
    """)

# ──────────────── HELPER FUNCTIONS ──────────────── #
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)
    return text.lower().strip()

def make_prediction(product, brand, source, review):
    combined = f"Product: {product}, Brand: {brand}, Source: {source}, Review: {review}"
    cleaned = clean_text(combined)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]
    confidence = round(max(model.predict_proba(vec)[0]) * 100, 2)
    return prediction, confidence

# ──────────────── INPUT FORM ──────────────── #
st.title("🕵️‍♂️ Fake Product Review Detector")
st.markdown("Check if a product review is **genuine or fake** using AI!")

product = st.text_input("🛍 Product Name")
brand = st.text_input("🏷 Brand Name")
source = st.selectbox("🌐 Where did you see the review?",
                      ["Amazon", "Flipkart", "YouTube", "Instagram", "Twitter", "Other"])
review_text = st.text_area("✍ Review Text")

# ──────────────── PREDICT BUTTON ──────────────── #
if st.button("🔍 Check Review"):
    if not (product and brand and review_text):
        st.warning("⚠️ Please fill all fields before submitting.")
    else:
        prediction, confidence = make_prediction(product, brand, source, review_text)

        # Styled output
        if prediction == 1:
            st.markdown(f"""
            <div style='background-color:#d4edda; padding:15px; border-radius:10px; color:#155724;'>
                ✅ <strong>Genuine Review</strong><br>
                Confidence: {confidence}%
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='background-color:#f8d7da; padding:15px; border-radius:10px; color:#721c24;'>
                🚨 <strong>Fake Review</strong><br>
                Confidence: {confidence}%
            </div>
            """, unsafe_allow_html=True)
