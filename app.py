from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import tensorflow as tf

# Load models
vgg16_model = tf.keras.models.load_model('path_to_your_model/vgg16_emotion_model.h5')
resnet50_model = tf.keras.models.load_model('path_to_your_model/resnet50_emotion_model.h5')
inceptionv3_model = tf.keras.models.load_model('path_to_your_model/inceptionv3_emotion_model.h5')
custom_model = tf.keras.models.load_model('path_to_your_model/custom_emotion_model.h5')

models = {
    'vgg16': vgg16_model,
    'resnet50': resnet50_model,
    'inceptionv3': inceptionv3_model,
    'custom': custom_model
}

app = Flask(__name__)
CORS(app)

def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (48, 48))
    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    model_name = data['model']
    image_data = np.frombuffer(bytes.fromhex(data['image']), dtype=np.uint8)
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    preprocessed_image = preprocess_image(image)
    
    model = models[model_name]
    predictions = model.predict(preprocessed_image)
    emotion_index = np.argmax(predictions[0])
    emotion = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise'][emotion_index]
    
    return jsonify({'emotion': emotion})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
