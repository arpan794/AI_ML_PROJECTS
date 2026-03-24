import string
import pandas as pd

def load_captions(path):

    captions = {}

    with open(path) as f:

        for line in f:

            tokens = line.strip().split(",")

            image_id = tokens[0]

            caption = tokens[1]

            captions.setdefault(image_id, []).append(caption)

    return captions


def clean_caption(caption):

    caption = caption.lower()

    caption = caption.translate(
        str.maketrans("", "", string.punctuation)
    )

    caption = "startseq " + caption + " endseq"

    return caption


def clean_captions(mapping):

    for key, caps in mapping.items():

        mapping[key] = [clean_caption(c) for c in caps]

    return mapping