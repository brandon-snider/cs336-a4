import fasttext

QUALITY_MODEL_PATH = "out/models/quality.bin"

quality_model = fasttext.load_model(QUALITY_MODEL_PATH)


label_map = {
    "positive": "wiki",
    "negative": "cc",
}


def classify_quality(text: str) -> tuple[bool, float]:
    """
    Classify a document as high quality or not.
    Returns a tuple of (is_high_quality, confidence)
    """
    clean = text.replace("\n", " ")
    labels, probs = quality_model.predict(clean, k=1)

    label = labels[0].replace("__label__", "")
    return label_map[label], probs[0]
