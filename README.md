# Emotion Classifier Model (CP468 Final Project)

![Demo Image](test_images/demo_pic.png)

## Kaggle Dataset
[CK+ Dataset](https://www.kaggle.com/datasets/shuvoalok/ck-dataset)

## Emotion Classes
The model classifies the following 7 emotions:
- Anger
- Contempt
- Disgust
- Fear
- Sadness
- Happy
- Surprise

## Pre-Trained Models used in Phase Two
- VGG16 Model
- ResNet50 Model
- InceptionV3 Model

## Model Training
To review the model training process, refer to `colab.py` where the dataset is pre-processed, and a CNN is trained from scratch. This model is evaluated against an ensemble of 3 pre-trained models through various metrics like True Positive (TP), True Negative (TN), False Positive (FP), and False Negative (FN). 

Results can be shown via graphs and manual tests.

## Demo
To demo the model, please refer to `app.py`, which builds a Gradio app and integrates with Spotify to provide a preset list of playlists pertaining to each emotion.

*Note: Install dependencies using `requirements.txt`.

## Authors
- Zaki
- Ved
- Tony
- Haseeb
