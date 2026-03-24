import numpy as np
import cv2

from .feature_extractor import FeatureExtractor
from .inference import CaptionGenerator


class ImageCaptionPipeline:

    def __init__(self):

        self.extractor = FeatureExtractor()

        self.generator = CaptionGenerator()

    def run(self, image_bytes):

        file_bytes = np.asarray(
            bytearray(image_bytes),
            dtype=np.uint8
        )

        image = cv2.imdecode(file_bytes, 1)

        features = self.extractor.extract(image)

        caption = self.generator.generate(features)

        return caption