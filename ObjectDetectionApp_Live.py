import streamlit as st
import tempfile
import cv2
import random
import numpy as np
from ultralytics import YOLO
from PIL import Image

# Configure Streamlit app
st.set_page_config(page_title="🎥 Object Detection App 🚀", layout="wide", page_icon="🤖")

# Add background image via CSS
st.markdown("""
    <style>
        .stApp {
            background-image: url("https://c4.wallpaperflare.com/wallpaper/223/48/930/plexus-network-abstract-line-art-connectivity-hd-wallpaper-preview.jpg");
            background-size: cover;
            background-position: center center;
            background-attachment: fixed;
        }
        .main-heading {
            font-size: 2.5rem;
            color: #ffffff;
            text-shadow: 2px 2px 5px #000000;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# Main heading
st.markdown('<h1 class="main-heading">🎥 Welcome to the Object Detection App 🚀</h1>', unsafe_allow_html=True)

# Sidebar setup
st.sidebar.title("🔧 App Options")
st.sidebar.markdown("""
### Welcome to the **Object Detection App**!
This app uses **YOLOv8** for detecting objects in:
- **Videos** 🎥
- **Images** 🖼️
- **Webcam** 📷

### How to Use:
1. Choose an option: Video, Image, or Webcam.
2. Upload the file or start the webcam.
3. Click **Detect** to see results.
4. Stop the detection or download the output when done.

Enjoy exploring object detection with cutting-edge AI!  
""")

# Dropdown selector for detection modes
option = st.selectbox(
    "Select Detection Mode:",
    ["None (App Details)", "Video Detection", "Image Detection", "Webcam Detection"],
    index=0,
    help="Select a mode to start detecting objects."
)

# Load YOLO model
model = YOLO("weights/yolov8n.pt")

# Load class labels
with open("coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

# Generate random colors for classes
detection_colors = [
    (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    for _ in range(len(class_list))
]

# Detection Function
def detect_objects(frame):
    results = model.predict(source=[frame], conf=0.45, save=False)
    boxes = results[0].boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy.numpy()[0])
        cls = int(box.cls.numpy()[0])
        conf = box.conf.numpy()[0]
        color = detection_colors[cls % len(detection_colors)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        label = f"{class_list[cls]} ({conf:.2f})"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame

# App Details when no option is selected
if option == "None (App Details)":
    st.subheader("🧐 App Overview:")
    st.write("""
    Welcome to the **Object Detection App**! This app allows you to explore state-of-the-art object detection using the **YOLOv8** model. 
    It supports video, image, and webcam-based detection and provides interactive outputs and download options.
    """)

    st.write("""
    #### Key Features:
    - **High Accuracy**: Powered by YOLOv8, a cutting-edge object detection model.
    - **Multi-format Support**: Detect objects in videos, images, and webcam streams.
    - **Interactive Results**: Download processed results for further use.
    - **Visually Engaging**: Enjoy a sleek, modern UI.
    """)

    st.info("Select a detection mode from the dropdown menu above to get started!")

# Video Detection
elif option == "Video Detection":
    st.title("🎥 Video Object Detection")
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_file:
        if st.button("🚀 Detect Objects in Video"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
                temp_video.write(uploaded_file.read())
                temp_video_path = temp_video.name

            cap = cv2.VideoCapture(temp_video_path)
            output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_file.name, fourcc, 20.0, (frame_width, frame_height))

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                detected_frame = detect_objects(frame)
                out.write(detected_frame)

            cap.release()
            out.release()

            st.video(output_file.name)
            with open(output_file.name, "rb") as file:
                st.download_button(
                    label="📥 Download Detected Video",
                    data=file,
                    file_name="detected_video.mp4",
                    mime="video/mp4",
                )

# Image Detection
elif option == "Image Detection":
    st.title("🖼️ Image Object Detection")
    uploaded_image = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        if st.button("🚀 Detect Objects in Image"):
            frame = np.array(image.convert("RGB"))
            detected_frame = detect_objects(frame)
            st.image(detected_frame, caption="Detected Image", use_column_width=True)
            detected_image_path = "detected_image.jpg"
            cv2.imwrite(detected_image_path, detected_frame)
            with open(detected_image_path, "rb") as file:
                st.download_button(
                    label="📥 Download Detected Image",
                    data=file,
                    file_name="detected_image.jpg",
                    mime="image/jpeg",
                )

# Webcam Detection
elif option == "Webcam Detection":
    st.title("📷 Webcam Object Detection")
    start_detection = st.button("🚀 Start Webcam Detection")
    if start_detection:
        cap = cv2.VideoCapture(0)
        FRAME_WINDOW = st.image([])
        stop_detection = st.button("🛑 Stop Detection")
        while cap.isOpened() and not stop_detection:
            ret, frame = cap.read()
            if not ret:
                st.error("Error: Failed to access webcam.")
                break
            detected_frame = detect_objects(frame)
            FRAME_WINDOW.image(cv2.cvtColor(detected_frame, cv2.COLOR_BGR2RGB))
        cap.release()
        cv2.destroyAllWindows()
        
# Footer with copyright notice
st.markdown("""
    <style>
        .footer {
            position: fixed;
            bottom: 0;
            width: 100%;
            text-align: center;
            color: #ffffff;
            background-color: rgba(0, 0, 0, 0.7);
            padding: 5px;
            font-size: 0.9rem;
            box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.5);
        }
    </style>
    <div class="footer">
        © 2024 | Developed by <b>Darshanikanta</b>
    </div>
""", unsafe_allow_html=True)

