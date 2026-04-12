import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import joblib

# Load model & scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(
    page_title="Handwritten Digit Recognizer",
    page_icon="🖌",
    layout="centered"
)

st.title("🖌 Handwritten Digit Recognizer")

st.write("Draw a digit (0–9) below and click Predict")

# Canvas
canvas_result = st_canvas(
    fill_color="white",
    stroke_width=15,
    stroke_color="black",
    background_color="white",
    height=420,
    width=580,
    drawing_mode="freedraw",
    key="canvas",
)

# Predict button
if st.button("Predict"):

    # Check if user actually drew something
    if canvas_result.image_data is None:
        st.error("Please draw something first!")
    else:
        img = Image.fromarray(canvas_result.image_data.astype("uint8")).convert("L")
        img = img.resize((28, 28))

        img_array = np.array(img)

        # Check blank canvas (all white)
        if np.mean(img_array) > 250:
            st.error("Canvas is empty. Please draw a digit.")
        else:
            img_array = 255 - img_array
            img_array = img_array.reshape(1, -1)
            img_array = scaler.transform(img_array)

            prediction = model.predict(img_array)

            st.success(f"🎯 Prediction: {prediction[0]}")

# Clear button
if st.button("Clear"):
    st.rerun()
