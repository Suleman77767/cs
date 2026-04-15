import streamlit as st
import cv2
import numpy as np
import mediapipe as mp

# UI setup
st.set_page_config(page_title="AI Computer Vision", layout="wide")
st.title("🤖 Local AI Vision App")
st.markdown("Is app mein **OpenCV** aur **Mediapipe** use ho raha hai (No API Required).")

# Library testing
try:
    cv2_version = cv2.__version__
    st.sidebar.success(f"OpenCV Loaded: v{cv2_version}")
except Exception as e:
    st.sidebar.error("Error: OpenCV library load nahi ho saki.")

uploaded_file = st.file_uploader("Tasveer upload karein...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Processing image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Face Detection
    mp_face = mp.solutions.face_detection
    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(image_rgb)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                             int(bboxC.width * iw), int(bboxC.height * ih)
                cv2.rectangle(image_rgb, (x, y), (x+w, y+h), (0, 255, 0), 6)
            
            st.image(image_rgb, caption="AI Result", use_container_width=True)
            st.success(f"Detected {len(results.detections)} faces!")
        else:
            st.image(image_rgb, caption="No faces detected", use_container_width=True)
            st.info("Koi chehra nahi mila.")
