import argparse
import random

DEFAULT_TRAIN_POS_PATH = "data/wiki/train_positive.txt"
DEFAULT_TRAIN_NEG_PATH = "data/wiki/train_negative.txt"
DEFAULT_OUTPUT_PATH = "data/wiki/train_all.txt"


def main(
    train_pos_path: str,
    train_neg_path: str,
    n: int | None,
    n_positive: int | None,
    n_negative: int | None,
    output_path: str,
    seed: int,
):
    if n is not None:
        n_positive = n_negative = n

    random.seed(seed)
    positive_lines = []
    negative_lines = []

    with open(train_pos_path) as f:
        positive_lines.extend(f.readlines())

    if n_positive is not None:
        random.shuffle(positive_lines)
        positive_lines = positive_lines[:n_positive]

    with open(train_neg_path) as f:
        negative_lines.extend(f.readlines())

    if n_negative is not None:
        random.shuffle(negative_lines)
        negative_lines = negative_lines[:n_negative]

    lines = positive_lines + negative_lines
    random.shuffle(lines)

    with open(output_path, "w") as f:
        f.writelines(lines)

    print(f"Wrote {len(lines)} training examples to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-pos-path", default=DEFAULT_TRAIN_POS_PATH)
    parser.add_argument("--train-neg-path", default=DEFAULT_TRAIN_NEG_PATH)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--n-positive", type=int, default=None)
    parser.add_argument("--n-negative", type=int, default=None)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(
        args.train_pos_path, args.train_neg_path, args.n, args.n_positive, args.n_negative, args.output_path, args.seed
    )
