import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

st.title("AI Computer Vision App")

# Test karne ke liye ke library load hui ya nahi
st.write(f"OpenCV Version: {cv2.__version__}")

uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    # Convert to image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Display
    st.image(img_rgb, caption='Uploaded Image')
