import streamlit as st
import cv2
import numpy as np
import mediapipe as mp

# Page Configuration
st.set_page_config(page_title="AI Vision", layout="centered")

st.title("🤖 Local AI Computer Vision")
st.markdown("Is application mein hum **Mediapipe** aur **OpenCV-Headless** use kar rahe hain.")

# File Uploader
uploaded_file = st.file_uploader("Apni tasveer select karein...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Processing image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Face Detection Model
    mp_face_detection = mp.solutions.face_detection
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(image_rgb)

        if results.detections:
            for detection in results.detections:
                # Drawing Bounding Box
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                             int(bboxC.width * iw), int(bboxC.height * ih)
                cv2.rectangle(image_rgb, (x, y), (x+w, y+h), (0, 255, 0), 10)
            
            st.image(image_rgb, caption="AI Result", use_container_width=True)
            st.success(f"Detected {len(results.detections)} face(s)!")
        else:
            st.image(image_rgb, caption="No faces found", use_container_width=True)
            st.warning("Koi chehra nahi mila.")
else:
    st.info("Tasveer upload karne ka intezar hai...")
