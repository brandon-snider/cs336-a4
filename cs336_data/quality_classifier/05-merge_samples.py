import argparse
import random

DEFAULT_TRAIN_POS_PATH = "data/wiki/train_positive.txt"
DEFAULT_TRAIN_NEG_PATH = "data/wiki/train_negative.txt"
DEFAULT_OUTPUT_PATH = "data/wiki/train_all.txt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-pos-path", default=DEFAULT_TRAIN_POS_PATH)
    parser.add_argument("--train-neg-path", default=DEFAULT_TRAIN_NEG_PATH)
    parser.add_argument("-n", type=int, default=None)
    parser.add_argument("--n-positive", type=int, default=None)
    parser.add_argument("--n-negative", type=int, default=None)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--seed", type=int, default=40)
    args = parser.parse_args()

    if args.n is not None:
        args.n_positive = args.n_negative = args.n

    random.seed(args.seed)
    positive_lines = []
    negative_lines = []

    with open(args.train_pos_path) as f:
        positive_lines.extend(f.readlines())

    if args.n_positive is not None:
        random.shuffle(positive_lines)
        positive_lines = positive_lines[: args.n_positive]

    with open(args.train_neg_path) as f:
        negative_lines.extend(f.readlines())

    if args.n_negative is not None:
        random.shuffle(negative_lines)
        negative_lines = negative_lines[: args.n_negative]

    lines = positive_lines + negative_lines
    random.shuffle(lines)

    with open(args.output_path, "w") as f:
        f.writelines(lines)

    print(f"Wrote {len(lines)} training examples to {args.output_path}")


if __name__ == "__main__":
    main()
