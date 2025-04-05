import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageOps

# Load the trained model
model = load_model("brain_tumor_model.h5")  # replace with your model name

# App title and description
st.set_page_config(page_title="Brain Tumor Detection")
st.title("🧠 Brain Tumor Detection")
st.write("Upload an MRI image to check for the presence of a brain tumor.")

# Upload image
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('L')  # convert to grayscale
    img = ImageOps.fit(img, (128, 128), Image.Resampling.LANCZOS)
    st.image(img, caption='Uploaded MRI Image', use_container_width=True)

    # Preprocess the image
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # shape becomes (1, 128, 128, 1)
    img_array = img_array / 255.0  # normalize

    # Predict
    prediction = model.predict(img_array)

    if prediction[0][0] > 0.5:
        st.error("Prediction: Brain Tumor Detected")
    else:
        st.success("Prediction: No Brain Tumor Detected")
