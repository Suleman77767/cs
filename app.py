import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

# Page setup
st.set_page_config(page_title="AI Computer Vision", layout="centered")
st.title("🤖 AI Computer Vision (No API)")

# Error handling for libraries
try:
    st.sidebar.success(f"OpenCV Version: {cv2.__version__}")
except Exception as e:
    st.error("OpenCV load nahi ho saki. Please check requirements.txt")

# Main Logic
uploaded_file = st.file_uploader("Tasveer upload karein", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Convert image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Face Detection Model (Pre-trained)
    mp_face = mp.solutions.face_detection
    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(frame_rgb)
        
        if results.detections:
            for detection in results.detections:
                # Draw Box
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                             int(bboxC.width * iw), int(bboxC.height * ih)
                cv2.rectangle(frame_rgb, (x, y), (x+w, y+h), (0, 255, 0), 4)
            
            st.image(frame_rgb, caption="AI Detection Result", use_container_width=True)
            st.success(f"Found {len(results.detections)} faces!")
        else:
            st.image(frame_rgb, caption="No faces detected", use_container_width=True)
