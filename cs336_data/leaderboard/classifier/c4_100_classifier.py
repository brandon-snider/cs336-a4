import fasttext

QUALITY_MODEL_PATH = "/data/c-sniderb/a4-leaderboard/classifier/quality.bin"

quality_model = fasttext.load_model(QUALITY_MODEL_PATH)


def classify_c4_100(text: str) -> tuple[str, float]:
    """
    Classify a document as positive (resembling c4_100) or negative (resembling cc).
    Returns a tuple of (is_c4_100, confidence)
    """
    clean = text.replace("\n", " ")
    labels, probs = quality_model.predict(clean, k=1)

    label = labels[0].replace("__label__", "")
    return label, probs[0]
