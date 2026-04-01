import numpy as np
import pickle

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model


class CaptionGenerator:

    def __init__(self):

        self.model = load_model(
            "backend/artifacts/caption_model.h5"
        )

        self.tokenizer = pickle.load(
            open("backend/artifacts/tokenizer.pkl", "rb")
        )

        self.max_length = pickle.load(
            open("backend/artifacts/max_length.pkl", "rb")
        )

    def generate(self, feature):

        caption = "startseq"

        for _ in range(self.max_length):

            seq = self.tokenizer.texts_to_sequences([caption])[0]

            seq = pad_sequences(
                [seq],
                maxlen=self.max_length
            )

            yhat = self.model.predict([feature, seq], verbose=0)

            index = np.argmax(yhat)

            word = self.tokenizer.index_word.get(index)

            if word is None:
                break

            caption += " " + word

            if word == "endseq":
                break

        return caption.replace("startseq", "").replace("endseq", "")