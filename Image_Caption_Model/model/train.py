import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from caption_model import build_model

features = pickle.load(open("saved_models/features.pkl", "rb"))

captions = []
mapping = {}

with open("data/captions.txt") as f:
    for line in f:
        img, caption = line.strip().split(",", 1)
        caption = "startseq " + caption + " endseq"
        captions.append(caption)
        mapping.setdefault(img, []).append(caption)

from utils.tokenizer import create_tokenizer
tokenizer = create_tokenizer(captions)

vocab_size = len(tokenizer.word_index) + 1
max_length = max(len(c.split()) for c in captions)

model = build_model(vocab_size, max_length)

X1, X2, y = [], [], []

for img, caps in mapping.items():
    for cap in caps:
        seq = tokenizer.texts_to_sequences([cap])[0]
        for i in range(1, len(seq)):
            in_seq = pad_sequences([seq[:i]], maxlen=max_length)[0]
            out_seq = to_categorical([seq[i]], num_classes=vocab_size)[0]
            X1.append(features[img][0])
            X2.append(in_seq)
            y.append(out_seq)

model.fit([np.array(X1), np.array(X2)], np.array(y), epochs=10, batch_size=64)
model.save("saved_models/caption_model.h5")