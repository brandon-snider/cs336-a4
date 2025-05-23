import os

import fasttext

NSFW_MODEL_NAME = "dolma_fasttext_nsfw_jigsaw_model.bin"
TOXIC_MODEL_NAME = "dolma_fasttext_hatespeech_jigsaw_model.bin"

MODEL_PATHS = ["/data/classifiers", "../classifiers"]

NSFW_MODEL_PATH = ""
TOXIC_MODEL_PATH = ""

for path in MODEL_PATHS:
    if os.path.exists(os.path.join(path, NSFW_MODEL_NAME)):
        NSFW_MODEL_PATH = os.path.join(path, NSFW_MODEL_NAME)
    if os.path.exists(os.path.join(path, TOXIC_MODEL_NAME)):
        TOXIC_MODEL_PATH = os.path.join(path, TOXIC_MODEL_NAME)

if NSFW_MODEL_PATH == "" or TOXIC_MODEL_PATH == "":
    raise ValueError("NSFW or toxic model not found")

nsfw_model = fasttext.load_model(NSFW_MODEL_PATH)
toxic_model = fasttext.load_model(TOXIC_MODEL_PATH)


def classify_nsfw(text: str) -> tuple[bool, float]:
    """
    Classify text as NSFW or not.
    Returns a tuple of (is_nsfw, confidence)
    """
    clean = text.replace("\n", " ")
    labels, probs = nsfw_model.predict(clean, k=1)

    label = labels[0].replace("__label__", "")
    return label, probs[0]


def classify_toxic_speech(text: str) -> tuple[bool, float]:
    """
    Classify text as toxic speech or not.
    Returns a tuple of (is_toxic, confidence)
    """
    clean = text.replace("\n", " ")
    labels, probs = toxic_model.predict(clean, k=1)

    label = labels[0].replace("__label__", "")
    return label, probs[0]
