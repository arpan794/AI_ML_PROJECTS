import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model("saved_models/caption_model.h5")
tokenizer = pickle.load(open("saved_models/tokenizer.pkl", "rb"))
max_length = 34

def generate_caption(feature):
    caption = "startseq"

    for _ in range(max_length):
        seq = tokenizer.texts_to_sequences([caption])[0]
        seq = pad_sequences([seq], maxlen=max_length)
        yhat = model.predict([feature, seq], verbose=0)
        word = tokenizer.index_word[np.argmax(yhat)]
        caption += " " + word
        if word == "endseq":
            break

    return caption.replace("startseq", "").replace("endseq", "")