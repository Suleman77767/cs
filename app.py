import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

# Initialize Mediapipe
mp_face = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

st.set_page_config(page_title="AI Computer Vision", layout="centered")

st.title("📸 Fast AI Vision App")
st.write("Upload an image to detect faces instantly.")

uploaded_file = st.file_uploader("Tasveer muntakhib karein...", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    # Convert file to opencv image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)

    # Run Face Detection
    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(opencv_image)

        # Draw detections
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(opencv_image, detection)
            st.success(f"AI found {len(results.detections)} face(s)!")
        else:
            st.info("No faces detected.")

    # Display Result
    st.image(opencv_image, caption='Processed Image', use_container_width=True)

else:
    st.warning("Waiting for image upload...")
