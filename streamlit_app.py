import cv2
import numpy as np
import torch
import streamlit as st
from facenet_pytorch import InceptionResnetV1
import pandas as pd
from datetime import datetime
import os
from face_detector import FaceDetector
from face_recognizer import FaceRecognizer
from voice_manager import VoiceManager

@st.cache_resource
def load_models():
    face_detector = FaceDetector()
    face_recognizer = FaceRecognizer("models")
    voice_manager = VoiceManager() 
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(
        face_detector.device
    )
    return face_detector, face_recognizer, resnet, voice_manager


face_detector, face_recognizer, resnet, voice_manager = load_models()
 

def save_to_excel(name, confidence):
    """Append a record to attendance.xlsx (only for known names)."""
    if name == "UNKNOWN":
        return

    file_path = "attendance.xlsx"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    new_entry = {
        "Name": name,
        "Timestamp": timestamp,
    }

    if not os.path.exists(file_path):
        df = pd.DataFrame([new_entry])
        df.to_excel(file_path, index=False)
    else:
        df = pd.read_excel(file_path)
        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
        df.to_excel(file_path, index=False)

if "already_recognized_names" not in st.session_state:
    st.session_state["already_recognized_names"] = set()

if "frame_count" not in st.session_state:
    st.session_state["frame_count"] = 0

if "running" not in st.session_state:
    st.session_state["running"] = False


process_every_n_frames = 30


def recognize_faces_in_frame(frame_bgr):
    faces_info = []

    detected_faces = face_detector.detect_faces(frame_bgr)

    for face_data in detected_faces:
        embedding = face_detector.extract_face_embedding(
            face_data["face_region"], resnet
        )
        if embedding is None:
            continue

        name, confidence = face_recognizer.recognize(embedding)

        if name != "UNKNOWN":
            if name not in st.session_state["already_recognized_names"]:
                st.session_state["already_recognized_names"].add(name)
                print(f"Result: {name}")
                save_to_excel(name, confidence)
                voice_manager.play("marked")  
        else:
            print("Result: UNKNOWN (low confidence)")
            voice_manager.play("unknown")

        faces_info.append({
            "bbox": {
                "x1": int(face_data["bbox"][0]),
                "y1": int(face_data["bbox"][1]),
                "x2": int(face_data["bbox"][2]),
                "y2": int(face_data["bbox"][3]),
            },
            "name": name,
            "confidence": float(confidence),
        })

    return faces_info

st.title("Continuous Webcam Face Recognition")

start_button = st.button("Start camera")
stop_button = st.button("Stop camera")

if start_button:
    st.session_state["running"] = True
if stop_button:
    st.session_state["running"] = False

frame_placeholder = st.empty()
info_placeholder = st.empty()

cap = cv2.VideoCapture(0)

while st.session_state["running"]:
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to read from camera.")
        break

    st.session_state["frame_count"] += 1
    faces = []
    if st.session_state["frame_count"] % process_every_n_frames == 0:
        faces = recognize_faces_in_frame(frame)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame_rgb, channels="RGB")
    with info_placeholder.container():
        if faces:
            for f in faces:
                st.write(f"- {f['name']}")

cap.release()
