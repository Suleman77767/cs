import streamlit as st

# Error handling for imports
try:
    import cv2
    import mediapipe as mp
    import numpy as np
    LIB_LOADED = True
except ImportError as e:
    LIB_LOADED = False
    ERROR_MSG = str(e)

st.set_page_config(page_title="AI Computer Vision", layout="centered")

st.title("🤖 AI Computer Vision (Free)")

if not LIB_LOADED:
    st.error(f"Libraries load nahi ho saki: {ERROR_MSG}")
    st.info("Check karein ke aapki file ka naam 'requirements.txt' hai aur usmein 'opencv-python-headless' likha hai.")
else:
    st.success("AI Models Loaded Successfully!")
    
    uploaded_file = st.file_uploader("Tasveer upload karein...", type=['jpg', 'png', 'jpeg'])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Face Detection
        mp_face = mp.solutions.face_detection
        with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
            results = face_detection.process(img_rgb)
            
            if results.detections:
                for detection in results.detections:
                    mp.solutions.drawing_utils.draw_detection(img_rgb, detection)
                st.image(img_rgb, caption="AI Result", use_container_width=True)
                st.balloons()
            else:
                st.image(img_rgb, caption="No face detected", use_container_width=True)
