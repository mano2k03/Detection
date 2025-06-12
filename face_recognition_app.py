import solara
import os
import cv2
import numpy as np
from pathlib import Path
import base64
import face_recognition

# Prepare dataset directory
DATASET_DIR = Path("dataset")
DATASET_DIR.mkdir(exist_ok=True)

# Reactive states
person_name = solara.reactive("")
status = solara.reactive("")
image_preview = solara.reactive(None)
recognition_result = solara.reactive("")
cropped_faces = solara.reactive([])

# Internal buffer (not reactive to avoid rerenders)
current_image = [None]
num_faces_value = [1]  # default

@solara.component
def Page():
    solara.Title("🧠 Face Detection Dataset App")

    solara.Text("👤 Step 1: Enter person's name")
    solara.InputText(label="Name", value=person_name)

    solara.Text("🧮 Step 2: Number of Faces to Save")
    solara.InputText(
        label="How many faces?",
        value=str(num_faces_value[0]),
        on_value=lambda v: update_num_faces(v)
    )

    solara.Text("📸 Step 3: Upload or Capture")
    solara.FileDrop(label="Drop JPG/PNG", on_file=on_file_upload)

    if solara.Button("📷 Capture from Webcam"):
        capture_from_webcam()

    if current_image[0] is not None and solara.Button("✅ Detect and Save Faces"):
        detect_and_save_faces()

    solara.Text(status.value)

    if image_preview.value:
        solara.Image(image_preview.value, format="jpeg", width="500px")

    if cropped_faces.value and solara.Button("🔍 Start Face Recognition"):
        start_face_recognition()

    if recognition_result.value:
        solara.Markdown(f"### 🎯 Recognition Result\n{recognition_result.value}")


def update_num_faces(v):
    try:
        n = int(v)
        if 1 <= n <= 10:
            num_faces_value[0] = n
            status.set("")
        else:
            status.set("❌ Enter a number between 1 and 10")
    except:
        status.set("❌ Invalid number")


def on_file_upload(file):
    data = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    current_image[0] = img
    preview_image(img)


def capture_from_webcam():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if ret:
        current_image[0] = frame
        preview_image(frame)
    else:
        status.set("❌ Webcam not accessible")


def preview_image(img):
    _, buffer = cv2.imencode(".jpg", img)
    base64_img = base64.b64encode(buffer).decode("utf-8")
    image_preview.set(f"data:image/jpeg;base64,{base64_img}")


def detect_and_save_faces():
    img = current_image[0]
    if img is None or not person_name.value:
        status.set("❌ Provide name and image first")
        return

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb)

    if not locations:
        status.set("❌ No faces found")
        return

    person_dir = DATASET_DIR / person_name.value
    person_dir.mkdir(parents=True, exist_ok=True)

    saved_faces = []
    for i, (top, right, bottom, left) in enumerate(locations[:num_faces_value[0]]):
        face_img = img[top:bottom, left:right]
        save_path = person_dir / f"face_{len(list(person_dir.glob('*.jpg')))+i}.jpg"
        cv2.imwrite(str(save_path), face_img)
        saved_faces.append(face_img)
        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)

    preview_image(img)
    cropped_faces.set(saved_faces)
    status.set(f"✅ Saved {len(saved_faces)} face(s) for {person_name.value}")


def start_face_recognition():
    known_encodings = []
    known_names = []

    for person_dir in DATASET_DIR.iterdir():
        if person_dir.is_dir():
            for img_path in person_dir.glob("*.jpg"):
                img = face_recognition.load_image_file(str(img_path))
                encs = face_recognition.face_encodings(img)
                if encs:
                    known_encodings.append(encs[0])
                    known_names.append(person_dir.name)

    results = []
    for face_img in cropped_faces.value:
        rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        encs = face_recognition.face_encodings(rgb_face)
        if encs:
            match = face_recognition.compare_faces(known_encodings, encs[0])
            name = "Unknown"
            if True in match:
                name = known_names[match.index(True)]
            results.append(name)
        else:
            results.append("No encoding")

    final_result = "\n".join([f"Face {i+1}: {name}" for i, name in enumerate(results)])
    recognition_result.set(final_result)
