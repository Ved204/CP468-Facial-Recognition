import gradio as gr

from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np

# parameters for loading data and images
detection_model_path = "config/face-detector.xml"
emotion_model_path = "models/ensemble_model.hdf5"

# hyper-parameters for bounding boxes shape
# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["anger", "disgust", "fear", "happy", "sadness", "surprise", "contempt"]

# Spotify playlists for each emotion
PLAYLISTS = {
    "anger": "https://open.spotify.com/playlist/37i9dQZF1EIgNZCaOGb0Mi?si=bcf5f8caabfa42d9",
    "disgust": "https://open.spotify.com/playlist/37i9dQZF1EIhuCNl2WSFYd?si=e9cc4a4ce9a2483d",
    "fear": "https://open.spotify.com/playlist/37i9dQZF1DWXe9gFZP0gtP?si=c6e4e9907e9f42e8",
    "happy": "https://open.spotify.com/playlist/37i9dQZF1EIgG2NEOhqsD7?si=03e929da14124808",
    "sadness": "https://open.spotify.com/playlist/37i9dQZF1EIdChYeHNDfK5?si=c3264add30164a4c",
    "surprise": "https://open.spotify.com/playlist/37i9dQZF1EIeU3RFfPV9ui?si=8057a396b44444c7",
    "contempt": "https://open.spotify.com/playlist/37i9dQZF1EIgau9in6RKng?si=d32e174b46f147d6",
}

def predict(frame):
    frame = imutils.resize(frame, width=300)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = face_detection.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    frameClone = frame.copy()
    if len(faces) > 0:
        faces = sorted(
            faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1])
        )[0]
        (fX, fY, fW, fH) = faces
        # Extract the ROI of the face from the grayscale image, resize it to a fixed 64x64 pixels, and then prepare
        # the ROI for classification via the CNN
        roi = gray[fY : fY + fH, fX : fX + fW]
        roi = cv2.resize(roi, (64, 64))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        preds = emotion_classifier.predict(roi)[0]
        label = EMOTIONS[preds.argmax()]
    else:
        return frameClone, "Can't find your face"

    probs = {}
    cv2.putText(
        frameClone, label, (fX, fY - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (238, 164, 64), 1
    )
    cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (238, 164, 64), 2)

    for i, (emotion, prob) in enumerate(zip(EMOTIONS, preds)):
        probs[emotion] = float(prob)

    playlist_url = PLAYLISTS[label]
    html_link = f'<a href="{playlist_url}" target="_blank">Listen to {label.capitalize()} Playlist</a>'

    return frameClone, html_link

inp = gr.components.Image(source="webcam", label="Your face", type="numpy")
out_image = gr.components.Image(label="Predicted Emotion", type="numpy")
out_html = gr.components.HTML(label="Spotify Playlist")

title = "Emotion Classification Model"
description = """
<div style="text-align: center;">
    Take a picture with your webcam, and it will guess if you are: happy, sadness, anger, disgust, fear, surprise, or contempt.
</div>
"""
thumbnail = "test_images/demo_pic.png"

interface = gr.Interface(
    fn=predict,
    inputs=inp,
    outputs=[out_image, out_html],
    capture_session=True,
    title=title,
    thumbnail=thumbnail,
    description=description,
)

interface.launch(inbrowser=True)
