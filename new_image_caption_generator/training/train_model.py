import pickle
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.models import Model


features = pickle.load(open("features.pkl", "rb"))


def load_captions():

    mapping = {}

    with open("data/captions.txt") as f:

        for line in f:

            img, caption = line.strip().split(",")

            caption = "startseq " + caption + " endseq"

            mapping.setdefault(img, []).append(caption)

    return mapping


mapping = load_captions()

captions = [c for caps in mapping.values() for c in caps]


tokenizer = Tokenizer()

tokenizer.fit_on_texts(captions)

vocab_size = len(tokenizer.word_index) + 1

max_length = max(len(c.split()) for c in captions)


def create_sequences():

    X1, X2, y = [], [], []

    for img, caps in mapping.items():

        for cap in caps:

            seq = tokenizer.texts_to_sequences([cap])[0]

            for i in range(1, len(seq)):

                in_seq = pad_sequences(
                    [seq[:i]],
                    maxlen=max_length
                )[0]

                out_seq = to_categorical(
                    seq[i],
                    num_classes=vocab_size
                )

                X1.append(features[img][0])

                X2.append(in_seq)

                y.append(out_seq)

    return np.array(X1), np.array(X2), np.array(y)


X1, X2, y = create_sequences()


# MODEL

inputs1 = Input(shape=(2048,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation="relu")(fe1)

inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(256)(se2)

decoder = add([fe2, se3])

decoder = Dense(256, activation="relu")(decoder)

outputs = Dense(vocab_size, activation="softmax")(decoder)

model = Model(inputs=[inputs1, inputs2], outputs=outputs)

model.compile(
    loss="categorical_crossentropy",
    optimizer="adam"
)

model.fit([X1, X2], y, epochs=10, batch_size=64)

model.save("backend/artifacts/caption_model.h5")

pickle.dump(
    tokenizer,
    open("backend/artifacts/tokenizer.pkl", "wb")
)

pickle.dump(
    max_length,
    open("backend/artifacts/max_length.pkl", "wb")
)