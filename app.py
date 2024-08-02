import gradio as gr

from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np

# parameters for loading data and images
detection_model_path = 'config/face-detector.xml'
emotion_model_path = 'models/ensemble_model.hdf5'

# hyper-parameters for bounding boxes shape
# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry", "disgusted", "scared", "happy", "sad", "surprised", "contempt"]

def predict(frame):
    frame = imutils.resize(frame, width=300)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = face_detection.detectMultiScale(gray, scaleFactor=1.1,
                                            minNeighbors=5, minSize=(30, 30),
                                            flags=cv2.CASCADE_SCALE_IMAGE)

    frameClone = frame.copy()
    if len(faces) > 0:
        faces = sorted(faces, reverse=True,
                       key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces
        # Extract the ROI of the face from the grayscale image, resize it to a fixed 64x64 pixels, and then prepare
        # the ROI for classification via the CNN
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (64, 64))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        preds = emotion_classifier.predict(roi)[0]
        label = EMOTIONS[preds.argmax()]
    else:
        return frameClone, "Can't find your face"

    probs = {}
    cv2.putText(frameClone, label, (fX, fY - 10),
                cv2.FONT_HERSHEY_DUPLEX, 1, (238, 164, 64), 1)
    cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                  (238, 164, 64), 2)

    for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
        probs[emotion] = float(prob)

    return frameClone, probs

inp = gr.components.Image(source="webcam", label="Your face", type="numpy")
out_image = gr.components.Image(label="Predicted Emotion", type="numpy")
out_label = gr.components.Label(num_top_classes=3, label="Top 3 Probabilities")

title = "Emotion Classification Model"
description = "Take a picture with your webcam, and it will guess if you are: happy, sad, angry, disgusted, scared, surprised, or contempt."
thumbnail = "test_images/demo_pic.png"

gr.Interface(predict, inp, [out_image, out_label], capture_session=True, title=title, thumbnail=thumbnail, description=description).launch(inbrowser=True)
