import json
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from cs336_data.gopher_quality_filters import gopher_quality_filter
from cs336_data.language_identification import identify_language

VAL_TEXT_PATH = "/data/c-sniderb/a4-leaderboard/paloma_c4_100_domains_validation_text.txt"


def main():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    data_path = "/data/paloma/tokenized_paloma_c4_100_domains_validation.npy"
    tokens = np.memmap(data_path, dtype=np.uint16, mode="r")

    eos = tokenizer.eos_token_id
    boundaries = np.where(tokens == eos)[0]
    docs = np.split(tokens, boundaries + 1)

    total_docs = 0
    english_docs = 0

    non_english_docs = []

    for doc in docs:
        text = tokenizer.decode(doc)
        total_docs += 1
        lang, score = identify_language(text)
        if lang == "en" and score > 0.75:
            english_docs += 1
        else:
            non_english_docs.append(
                {
                    "lang": lang,
                    "score": score,
                    "text": text,
                }
            )

    json.dump(non_english_docs, open("non_english_docs.json", "w"), indent=4)

    print(f"Total snippets: {total_docs}")
    print(f"English snippets: {english_docs}")
    print(f"English snippets percentage: {english_docs / total_docs:.2%}")


def check_english_of_val():
    with open(VAL_TEXT_PATH, "r") as f:
        docs = f.read().split("\n\n---END_OF_DOC---\n\n")

    english_docs = 0
    gopher_quality_docs = 0
    english_gopher_quality_docs = 0

    for doc in tqdm(docs):
        is_english = False
        is_gopher_quality = False

        lang, score = identify_language(doc)
        if lang == "en" and score > 0.9:
            is_english = True
            english_docs += 1

        is_gopher_quality = gopher_quality_filter(doc)
        if is_gopher_quality:
            gopher_quality_docs += 1
            is_gopher_quality = True

        if is_english and is_gopher_quality:
            english_gopher_quality_docs += 1

    print(f"Total snippets: {len(docs)}")
    print(f"English snippets: {english_docs}")
    print(f"English snippets percentage: {english_docs / len(docs):.2%}")
    print(f"Gopher quality snippets: {gopher_quality_docs}")
    print(f"Gopher quality snippets percentage: {gopher_quality_docs / len(docs):.2%}")
    print(f"English gopher quality snippets: {english_gopher_quality_docs}")
    print(f"English gopher quality snippets percentage: {english_gopher_quality_docs / len(docs):.2%}")


if __name__ == "__main__":
    # main()
    check_english_of_val()
