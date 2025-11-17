import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os
from streamlit_image_comparison import image_comparison

# ------------------------------------------------
# FACE CASCADE LOADER
# ------------------------------------------------
def load_face_cascade():
    local_path_alt2 = os.path.join("models", "haarcascade_frontalface_alt2.xml")
    local_path_default = os.path.join("models", "haarcascade_frontalface_default.xml")

    if os.path.exists(local_path_alt2):
        cascade = cv2.CascadeClassifier(local_path_alt2)
    elif os.path.exists(local_path_default):
        cascade = cv2.CascadeClassifier(local_path_default)
    else:
        st.error("Haarcascade XML not found in models/. Add haarcascade_frontalface_alt2.xml or haarcascade_frontalface_default.xml")
        return None
    return cascade

# ------------------------------------------------
# PIXELATE FUNCTION
# ------------------------------------------------
def pixelate_region(img, x, y, w, h, level=10):
    roi = img[y:y+h, x:x+w]
    level = max(1, int(level))
    small = cv2.resize(roi, (max(1, w // level), max(1, h // level)), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    img[y:y+h, x:x+w] = pixelated
    return img

# ------------------------------------------------
# BLUR FUNCTION
# ------------------------------------------------
def blur_region(img, x, y, w, h, level=25):
    roi = img[y:y+h, x:x+w]
    k = max(1, (int(level) // 2) * 2 + 1)
    blurred = cv2.GaussianBlur(roi, (k, k), 0)
    img[y:y+h, x:x+w] = blurred
    return img

# ------------------------------------------------
# FACE DETECTION WITH RESIZING
# ------------------------------------------------
def detect_faces(gray, cascade, scale_factor=1.1, min_neighbors=5):
    if cascade is None:
        return []

    h_orig, w_orig = gray.shape
    scale = 600 / max(h_orig, w_orig)
    if scale < 1.0:
        gray_resized = cv2.resize(gray, (int(w_orig*scale), int(h_orig*scale)))
    else:
        gray_resized = gray.copy()

    faces = cascade.detectMultiScale(
        gray_resized,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors
    )

    if scale < 1.0:
        faces = [(int(x/scale), int(y/scale), int(w/scale), int(h/scale)) for (x,y,w,h) in faces]

    return faces

# ------------------------------------------------
# STREAMLIT UI
# ------------------------------------------------
st.set_page_config(page_title="Face Blur Demo", layout="wide")
st.title("Face Blur / Pixelate Demo (Live Multi-Face Preview)")
st.markdown("Upload an image, choose blur or pixelation, and adjust sliders for live face detection preview!")

uploaded = st.file_uploader("Choose an image...", type=["jpg","jpeg","png","bmp","webp"])
mode = st.selectbox("Mode", ["blur", "pixelate"])
level = st.slider("Strength (higher = stronger)", min_value=1, max_value=80, value=25)
show_boxes = st.checkbox("Show detected face boxes (debug)")

# --------------------------
# Preset buttons for detection settings
# --------------------------
preset = st.radio("Detection Presets", ["Custom", "Fast Mode", "Accurate Mode"])
if preset == "Fast Mode":
    scale_factor, min_neighbors = 1.2, 3
elif preset == "Accurate Mode":
    scale_factor, min_neighbors = 1.05, 6
else:
    # Custom sliders
    scale_factor = st.slider("Face Detection Scale Factor", min_value=1.01, max_value=1.5, value=1.1, step=0.01)
    min_neighbors = st.slider("Face Detection Min Neighbors", min_value=1, max_value=10, value=5, step=1)

# ------------------------------------------------
# FUNCTION TO PROCESS IMAGE
# ------------------------------------------------
def process_image(image, cascade, scale_factor, min_neighbors, mode, level, show_boxes):
    img = np.array(image)[:, :, ::-1]  # RGB -> BGR
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray, cascade, scale_factor=scale_factor, min_neighbors=min_neighbors)
    result = img.copy()
    for (x, y, w, h) in faces:
        if mode == "blur":
            result = blur_region(result, x, y, w, h, level=level)
        else:
            result = pixelate_region(result, x, y, w, h, level=level)
        if show_boxes:
            cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return result[:, :, ::-1]  # Convert back to RGB

# ------------------------------------------------
# PROCESS IMAGE AND LIVE PREVIEW
# ------------------------------------------------
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    cascade = load_face_cascade()

    # Process image dynamically on slider change
    result_rgb = process_image(image, cascade, scale_factor, min_neighbors, mode, level, show_boxes)

    # Interactive before/after slider
    st.header("Before / After")
    image_comparison(
        img1=image,
        img2=result_rgb,
        label1="Original",
        label2="Processed",
        show_labels=True,
        width=800
    )

    # Download button
    buf = cv2.imencode(".jpg", np.array(result_rgb)[:, :, ::-1])[1].tobytes()
    st.download_button(
        "Download processed image",
        data=buf,
        file_name="processed.jpg",
        mime="image/jpeg",
    )
