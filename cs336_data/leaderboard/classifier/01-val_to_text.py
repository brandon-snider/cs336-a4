import numpy as np
import os
from transformers import AutoTokenizer
from tqdm import tqdm

data_dir = "/data/c-sniderb/a4-leaderboard"
data_path = os.path.join(data_dir, "tokenized_paloma_c4_100_domains_validation.bin")
out_path = os.path.join(data_dir, "paloma_c4_100_domains_validation_text.txt")
tokenizer = AutoTokenizer.from_pretrained("gpt2")


def main():
    tokens = np.memmap(data_path, dtype=np.uint16, mode="r")  # shape: (N,)
    eos = tokenizer.eos_token_id
    boundaries = np.where(tokens == eos)[0]
    docs = np.split(tokens, boundaries + 1)

    longest_doc = max(len(doc) for doc in docs)
    print(f"Longest doc: {longest_doc}")

    shortest_doc = min(len(doc) for doc in docs)
    print(f"Shortest doc: {shortest_doc}")

    mean_doc_length = sum(len(doc) for doc in docs) / len(docs)
    print(f"Mean doc length: {mean_doc_length}")

    with open(out_path, "w") as f:
        for doc in tqdm(docs, desc="Writing docs"):
            if len(doc) > 0 and doc[-1] == eos:
                doc = doc[:-1]
            if len(doc) == 0:
                continue
            f.write(tokenizer.decode(doc))
            f.write("\n\n---END_OF_DOC---\n\n")

    print(f"Total docs: {len(docs)}")
    print(f"Total tokens: {len(tokens)}")
    print("Special tokens:", tokenizer.special_tokens_map)


if __name__ == "__main__":
    main()
