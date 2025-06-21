import tensorflow as tf

# resave_image_model.py
from tensorflow.keras.models import load_model

# Load your existing .h5 model
model = load_model("model/product_auth_model.h5", compile=False)

# Save in new .keras format (recommended by TensorFlow)
model.save("model/product_auth_model.keras")
print("✅ Resaved model in .keras format")

