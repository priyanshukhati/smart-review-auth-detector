from keras.models import load_model, save_model

# Load your existing model saved with older TF/Keras
model = load_model("model/product_auth_model.h5", compile=False)

# Re-save it using Keras 3 format
model.save("model/product_auth_model.keras", save_format="keras")
