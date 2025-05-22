import random
import os
import argparse

DATA_DIR = "/data/c-sniderb/a4-leaderboard"
DATA_PATH = os.path.join(DATA_DIR, "paloma_c4_100_domains_validation_text.txt")
OUT_DIR = os.path.join(DATA_DIR, "classifier")
OUT_PATH_TRAIN = os.path.join(OUT_DIR, "positives_train.txt")
OUT_PATH_VALID = os.path.join(OUT_DIR, "positives_valid.txt")

NUM_TRAIN_EXAMPLES = 28000
NUM_VALID_EXAMPLES = 500


def main(
    data_path: str = DATA_PATH,
    out_dir: str = OUT_DIR,
    out_path_train: str = OUT_PATH_TRAIN,
    out_path_valid: str = OUT_PATH_VALID,
    num_train_examples: int = NUM_TRAIN_EXAMPLES,
    num_valid_examples: int = NUM_VALID_EXAMPLES,
):
    with open(data_path) as f:
        docs = f.read().split("\n\n---END_OF_DOC---\n\n")
    docs = [doc for doc in docs if doc.strip()]

    random.seed(42)
    random.shuffle(docs)

    valid_examples = docs[:num_valid_examples]
    train_examples = docs[num_valid_examples:]

    while num_train_examples > len(train_examples):
        train_examples.extend(train_examples)

    sampled_docs = train_examples[:num_train_examples]

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(out_path_train, "w") as f:
        for doc in sampled_docs:
            joined_text = doc.replace("\n", " ")
            f.write(f"__label__positive {joined_text}\n")

    print(f"Wrote {len(sampled_docs)} positive samples to {out_path_train}")

    with open(out_path_valid, "w") as f:
        for doc in valid_examples:
            joined_text = doc.replace("\n", " ")
            f.write(f"__label__positive {joined_text}\n")

    print(f"Wrote {len(valid_examples)} positive samples to {out_path_valid}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-train-examples", type=int, default=NUM_TRAIN_EXAMPLES)
    parser.add_argument("--num-valid-examples", type=int, default=NUM_VALID_EXAMPLES)
    args = parser.parse_args()
    main(num_train_examples=args.num_train_examples, num_valid_examples=args.num_valid_examples)
