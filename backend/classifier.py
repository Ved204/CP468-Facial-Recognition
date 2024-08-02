import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from PIL import ImageFile, Image
from numpy import expand_dims
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from custom_layers import CustomBatchNormalization

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Define the relative path to the custom model
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, '../models/resnet50_emotion_model.h5')

# Load the model with custom objects
custom_objects = {'BatchNormalization': CustomBatchNormalization}
model = load_model(model_path, custom_objects=custom_objects)

def getPrediction(img_bytes, model):
    # Loads the image and transforms it to (75, 75, 3) shape
    original_image = Image.open(img_bytes)
    original_image = original_image.convert('RGB')
    original_image = original_image.resize((75, 75), Image.NEAREST)
    
    numpy_image = image.img_to_array(original_image)
    image_batch = expand_dims(numpy_image, axis=0)
    
    # Normalize the image in the same way as during training
    processed_image = image_batch / 255.0
    preds = model.predict(processed_image)
    
    return preds

def classifyImage(file):
    # Returns a probability scores matrix 
    preds = getPrediction(file, model)
    
    # Pick the top prediction
    result = np.argmax(preds, axis=1)[0]
    return str(result)
