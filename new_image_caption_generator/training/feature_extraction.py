import os
import pickle
import cv2
import numpy as np

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import Model


def extract_features(image_dir):

    model = InceptionV3(weights="imagenet")

    model = Model(model.input, model.layers[-2].output)

    features = {}

    for img_name in os.listdir(image_dir):

        path = os.path.join(image_dir, img_name)

        img = cv2.imread(path)

        img = cv2.resize(img, (299, 299))

        img = preprocess_input(img)

        img = np.expand_dims(img, axis=0)

        feature = model.predict(img)

        features[img_name] = feature

    pickle.dump(features, open("features.pkl", "wb"))