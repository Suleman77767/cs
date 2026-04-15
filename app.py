import streamlit as st
import cv2
import numpy as np
import mediapipe as mp

# Page Config
st.set_page_config(page_title="AI Vision Pro", page_icon="🤖")

st.title("🤖 AI Computer Vision (No API)")
st.write("Upload an image for local Face Detection.")

# OpenCV Loading Check
try:
    ver = cv2.__version__
    st.sidebar.success(f"OpenCV v{ver} Loaded Successfully")
except Exception:
    st.sidebar.error("OpenCV not found. Check requirements.txt")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert uploaded file to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Initialize Mediapipe Face Detection
    mp_face = mp.solutions.face_detection
    
    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(image_rgb)

        if results.detections:
            for detection in results.detections:
                # Draw bounding box
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                             int(bboxC.width * iw), int(bboxC.height * ih)
                cv2.rectangle(image_rgb, (x, y), (x+w, y+h), (0, 255, 0), 8)
            
            st.image(image_rgb, caption="AI Result", use_container_width=True)
            st.success(f"Successfully detected {len(results.detections)} face(s)!")
        else:
            st.image(image_rgb, caption="Processed Image", use_container_width=True)
            st.warning("No faces detected in this image.")
else:
    st.info("Please upload an image to start detection.")
