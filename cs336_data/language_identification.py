import os
import fasttext

MODEL_PATH = (
    "/data/classifiers/lid.176.bin" if os.path.exists("/data/classifiers/lid.176.bin") else "../classifiers/lid.176.bin"
)

model = fasttext.load_model(MODEL_PATH)


def identify_language(text: str) -> str:
    clean = text.replace("\n", " ")
    labels, probs = model.predict(clean, k=1)

    lang = labels[0].replace("__label__", "")
    return lang, probs[0]
