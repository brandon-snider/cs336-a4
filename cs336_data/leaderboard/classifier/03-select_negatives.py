import argparse
import random
import os
from tqdm import tqdm

DATA_DIR = "/data/c-sniderb/a4-leaderboard/03-exact-deduped"
OUT_DIR = os.path.join(DATA_DIR, "..", "classifier")
OUT_PATH = os.path.join(OUT_DIR, "negatives_train.txt")
OUT_PATH_VALID = os.path.join(OUT_DIR, "negatives_valid.txt")

NUM_TRAIN_EXAMPLES = 28000
NUM_VALID_EXAMPLES = 500


def main(
    num_train_examples: int = NUM_TRAIN_EXAMPLES,
    num_valid_examples: int = NUM_VALID_EXAMPLES,
):
    data_filepaths = [os.path.join(DATA_DIR, filepath) for filepath in os.listdir(DATA_DIR)]

    examples_per_file = 0
    while examples_per_file * (examples_per_file * 10) < num_train_examples + num_valid_examples:
        examples_per_file += 1
    n_files = examples_per_file * 10

    random.seed(42)
    chosen_paths = random.sample(data_filepaths, n_files)
    examples = []

    for data_filepath in tqdm(chosen_paths, desc="Processing files"):
        with open(data_filepath) as f:
            docs = f.read().split("\n\n---END_OF_DOC---\n\n")
            docs = [doc for doc in docs if doc.strip()]
            chosen_docs = random.sample(docs, examples_per_file)
            examples.extend(chosen_docs)

    valid_examples = examples[:num_valid_examples]
    train_examples = examples[num_valid_examples:]

    if len(train_examples) > num_train_examples:
        train_examples = train_examples[:num_train_examples]

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    with open(OUT_PATH, "w") as f:
        for example in train_examples:
            joined_text = example.replace("\n", " ")
            f.write(f"__label__negative {joined_text}\n")

    print(f"Wrote {len(train_examples)} negative examples to {OUT_PATH}")

    with open(OUT_PATH_VALID, "w") as f:
        for example in valid_examples:
            joined_text = example.replace("\n", " ")
            f.write(f"__label__negative {joined_text}\n")

    print(f"Wrote {len(valid_examples)} negative examples to {OUT_PATH_VALID}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-train-examples", type=int, default=NUM_TRAIN_EXAMPLES)
    parser.add_argument("--num-valid-examples", type=int, default=NUM_VALID_EXAMPLES)
    args = parser.parse_args()
    main(num_train_examples=args.num_train_examples, num_valid_examples=args.num_valid_examples)
