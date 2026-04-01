import numpy as np
import cv2

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import Model


class FeatureExtractor:

    def __init__(self):

        base_model = InceptionV3(weights="imagenet")

        self.model = Model(
            base_model.input,
            base_model.layers[-2].output
        )

    def extract(self, image):

        image = cv2.resize(image, (299, 299))

        image = preprocess_input(image)

        image = np.expand_dims(image, axis=0)

        return self.model.predict(image)