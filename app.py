import streamlit as st
import pickle
import tensorflow as tf
import numpy as np
from PIL import Image

# --- Load Review Model and Vectorizer ---
with open("model/review_model.pkl", "rb") as f:
    review_model = pickle.load(f)

with open("model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# --- Load Image Classification Model ---
image_model = tf.keras.models.load_model("model/product_auth_model.h5", compile=False)

# --- Streamlit Page Config ---
st.set_page_config(page_title="🧠 Smart Product Review & Authenticity Detector", layout="wide")

st.markdown(
    "<h1 style='text-align: center; color: #4B8BBE;'>🧠 Smart Product Review & Authenticity Detector</h1>",
    unsafe_allow_html=True,
)

st.markdown(
    "<p style='text-align: center; font-size: 18px;'>Analyze product reviews for authenticity and verify product images to detect duplicates.</p>",
    unsafe_allow_html=True,
)

# --- Layout Columns (side by side) ---
left_col, right_col = st.columns(2)

# --- Left Column: Review Analysis ---
with left_col:
    st.markdown("## 📝 Review Authenticity Analysis")
    product_name = st.text_input("Product Name")
    brand_name = st.text_input("Brand Name")
    source = st.selectbox("Where did you see the review?", ["Amazon", "Flipkart", "Myntra", "Snapdeal", "Other"])
    review_text = st.text_area("Enter the Product Review")

    if st.button("Analyze Review"):
        if review_text.strip() == "":
            st.warning("Please enter a review to analyze.")
        else:
            vector = vectorizer.transform([review_text])
            pred_proba = review_model.predict_proba(vector)[0]
            label = "Genuine" if np.argmax(pred_proba) == 0 else "Fake"
            confidence = np.max(pred_proba) * 100

            if label == "Genuine":
                st.markdown(
                    f"<div style='padding: 12px; background-color: #e8f5e9; color: #2e7d32; "
                    f"border-radius: 8px; font-weight: bold;'>"
                    f"✅ Prediction: {label} Review (Confidence: {confidence:.2f}%)"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div style='padding: 12px; background-color: #ffebee; color: #c62828; "
                    f"border-radius: 8px; font-weight: bold;'>"
                    f"❌ Prediction: {label} Review (Confidence: {confidence:.2f}%)"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            with st.expander("🔍 See Prediction Probabilities"):
                st.markdown(f"- 🟢 Genuine confidence: `{pred_proba[0]*100:.2f}%`")
                st.markdown(f"- 🔴 Fake confidence: `{pred_proba[1]*100:.2f}%`")

# --- Right Column: Product Image Authenticity ---
with right_col:
    st.markdown("## 🖼️ Product Image Authenticity Check")
    uploaded_image = st.file_uploader("Upload a Product Image", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        img = Image.open(uploaded_image).convert("RGB")
        st.image(img, caption="Uploaded Product Image", use_container_width=True)

        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = image_model.predict(img_array)

        if prediction.shape[-1] == 1:
            confidence = prediction[0][0]
            label = "Original" if confidence >= 0.5 else "Duplicate"
            st.markdown(
                f"<div style='padding: 12px; background-color: {'#e8f5e9' if label == 'Original' else '#ffebee'}; "
                f"color: {'#2e7d32' if label == 'Original' else '#c62828'}; border-radius: 8px; font-weight: bold;'>"
                f"{'✅' if label == 'Original' else '❌'} Prediction: {label} Product (Confidence: {confidence*100:.2f}%)"
                f"</div>",
                unsafe_allow_html=True,
            )
        elif prediction.shape[-1] == 2:
            label = "Original" if np.argmax(prediction) == 1 else "Duplicate"
            confidence = np.max(prediction) * 100
            st.markdown(
                f"<div style='padding: 12px; background-color: {'#e8f5e9' if label == 'Original' else '#ffebee'}; "
                f"color: {'#2e7d32' if label == 'Original' else '#c62828'}; border-radius: 8px; font-weight: bold;'>"
                f"{'✅' if label == 'Original' else '❌'} Prediction: {label} Product (Confidence: {confidence:.2f}%)"
                f"</div>",
                unsafe_allow_html=True,
            )

            with st.expander("🔍 See Prediction Probabilities"):
                st.markdown(f"- 🟢 Original confidence: `{prediction[0][1]*100:.2f}%`")
                st.markdown(f"- 🔴 Duplicate confidence: `{prediction[0][0]*100:.2f}%`")
        else:
            st.error("⚠️ Model output not as expected. Please check your model's final layer configuration.")
