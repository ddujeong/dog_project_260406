import os
import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.preprocessing import normalize


MODEL_PATH = "models/body/detailed_body_classifier.keras"

LABEL_MAP = {
    0: "slim",
    1: "standard",
    2: "broad",
    3: "overweight",
}

model = None


def load_body_model():
    global model
    if model is None:
        model = tf.keras.models.load_model(MODEL_PATH)
    return model


def preprocess_image(image: Image.Image):
    img = image.convert("RGB").resize((224, 224))
    img_array = np.array(img).astype("float32")
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict_body_type(image: Image.Image):
    model = load_body_model()
    img_array = preprocess_image(image)

    preds = model.predict(img_array)[0]

    top2_idx = preds.argsort()[-2:][::-1]

    result = {
        "primary": LABEL_MAP[top2_idx[0]],
        "secondary": LABEL_MAP[top2_idx[1]],
        "primary_prob": float(preds[top2_idx[0]]),
        "secondary_prob": float(preds[top2_idx[1]]),
    }

    return result