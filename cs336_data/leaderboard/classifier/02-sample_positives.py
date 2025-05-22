import random
import os
import argparse

DATA_DIR = "/data/c-sniderb/a4-leaderboard"
DATA_PATH = os.path.join(DATA_DIR, "paloma_c4_100_domains_validation_text.txt")
OUT_DIR = os.path.join(DATA_DIR, "classifier")
OUT_PATH = os.path.join(OUT_DIR, "positives.txt")


def main(data_path: str = DATA_PATH, out_dir: str = OUT_DIR, out_path: str = OUT_PATH, num_examples: int = 14000):
    with open(data_path) as f:
        docs = f.read().split("\n\n---END_OF_DOC---\n\n")
    docs = [doc for doc in docs if doc.strip()]
    sampled_docs = random.sample(docs, num_examples)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(out_path, "w") as f:
        for doc in sampled_docs:
            joined_text = doc.replace("\n", " ")
            f.write(f"__label__positive {joined_text}\n")

    print(f"Wrote {len(sampled_docs)} positive samples to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-examples", type=int, default=14000)
    args = parser.parse_args()
    main(num_examples=args.num_examples)
