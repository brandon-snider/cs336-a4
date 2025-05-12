import argparse
import random

DEFAULT_TRAIN_POS_PATH = "data/wiki/train_positive.txt"
DEFAULT_TRAIN_NEG_PATH = "data/wiki/train_negative.txt"
DEFAULT_OUTPUT_PATH = "data/wiki/train_all.txt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-pos-path", default=DEFAULT_TRAIN_POS_PATH)
    parser.add_argument("--train-neg-path", default=DEFAULT_TRAIN_NEG_PATH)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--seed", type=int, default=40)
    args = parser.parse_args()

    lines = []

    with open(args.train_pos_path) as f:
        lines.extend(f.readlines())

    with open(args.train_neg_path) as f:
        lines.extend(f.readlines())

    random.seed(args.seed)
    random.shuffle(lines)

    with open(args.output_path, "w") as f:
        f.writelines(lines)

    print(f"Wrote {len(lines)} training examples to {args.output_path}")


if __name__ == "__main__":
    main()
