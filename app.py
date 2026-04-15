import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

# Mediapipe ki initialization (Local models)
mp_face_detection = mp.solutions.face_detection
mp_selfie_segmentation = mp.solutions.self_selfie_segmentation

# Page Configuration
st.set_page_config(page_title="AI Vision App", layout="wide")

st.title("🤖 AI Computer Vision Dashboard")
st.markdown("Is app mein hum **Pre-trained Models** use kar rhy hain jo bina kisi API key ke chalty hain.")

# Sidebar for Navigation
option = st.sidebar.selectbox(
    'Feature Select Karein:',
    ('Face Detection', 'Background Blur')
)

st.sidebar.info("Ye app Github aur Streamlit par deploy hone ke liye ready hai.")

# File Uploader
uploaded_file = st.file_uploader("Image upload karein...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Image ko process karne ke liye convert karna
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Original Image")
        st.image(image, use_container_width=True)

    with col2:
        st.header("AI Result")
        
        # FEATURE 1: FACE DETECTION
        if option == 'Face Detection':
            with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
                results = face_detection.process(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
                
                annotated_image = img_array.copy()
                if results.detections:
                    for detection in results.detections:
                        # Bounding box draw karna
                        bboxC = detection.location_data.relative_bounding_box
                        ih, iw, _ = annotated_image.shape
                        bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                               int(bboxC.width * iw), int(bboxC.height * ih)
                        cv2.rectangle(annotated_image, bbox, (0, 255, 0), 10)
                    
                    st.image(annotated_image, use_container_width=True)
                    st.success(f"AI ne {len(results.detections)} chehry detect kiye!")
                else:
                    st.warning("Koi chehra nahi mila.")

        # FEATURE 2: BACKGROUND BLUR
        elif option == 'Background Blur':
            with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_seg:
                # Background segmentation perform karna
                results = selfie_seg.process(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
                mask = results.segmentation_mask > 0.1
                
                # Background ko blur karna
                blurred_image = cv2.GaussianBlur(img_array, (55, 55), 0)
                
                # Mask apply karna (Condition ? Image : Blurred)
                output_image = np.where(mask[:, :, None], img_array, blurred_image)
                
                st.image(output_image, use_container_width=True)
                st.success("Background successfully blur ho gaya!")

else:
    st.info("Upar majood button se koi bhi tasveer select karein taake AI apna kaam dikha saky.")
