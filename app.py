import streamlit as st
import cv2
import numpy as np
import mediapipe as mp

st.set_page_config(page_title="AI Vision", layout="wide")

st.title("🤖 Free AI Computer Vision App")

# Library Check
try:
    st.sidebar.success(f"✅ OpenCV Loaded (v{cv2.__version__})")
except Exception as e:
    st.sidebar.error("❌ OpenCV load nahi ho saki")

# User Friendly UI
st.info("Bina kisi API ke Computer Vision ka istemal karein.")

uploaded_file = st.file_uploader("Tasveer muntakhib karein...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Convert image to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Pre-trained Face Detection Model
    mp_face = mp.solutions.face_detection
    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(image_rgb)

        if results.detections:
            for detection in results.detections:
                # Drawing Bounding Box
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                             int(bboxC.width * iw), int(bboxC.height * ih)
                cv2.rectangle(image_rgb, (x, y), (x+w, y+h), (0, 255, 0), 5)
            
            st.image(image_rgb, caption="AI Result: Face Detected", use_container_width=True)
            st.success(f"Total Faces Found: {len(results.detections)}")
        else:
            st.image(image_rgb, caption="No face detected", use_container_width=True)
            st.warning("Koi chehra nahi mila.")
