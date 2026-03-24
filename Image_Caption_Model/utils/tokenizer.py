from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

def create_tokenizer(captions):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(captions)
    pickle.dump(tokenizer, open("saved_models/tokenizer.pkl", "wb"))
    return tokenizer