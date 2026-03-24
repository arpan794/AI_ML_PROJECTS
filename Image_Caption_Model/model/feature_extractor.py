import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model
import numpy as np
import cv2
import os
import pickle

def extract_features(image_dir):
    base_model = InceptionV3(weights='imagenet')
    model = Model(base_model.input, base_model.layers[-2].output)

    features = {}

    for img_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (299, 299))
        img = preprocess_input(img)
        img = np.expand_dims(img, axis=0)

        feature = model.predict(img, verbose=0)
        features[img_name] = feature

    pickle.dump(features, open("saved_models/features.pkl", "wb"))