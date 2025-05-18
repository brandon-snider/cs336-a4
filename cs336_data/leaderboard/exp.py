import json
import numpy as np
from transformers import AutoTokenizer
from cs336_data.language_identification import identify_language


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


if __name__ == "__main__":
    main()
