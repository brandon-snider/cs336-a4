import argparse

DEFAULT_INPUT_PATH = "data/wiki/train_all.txt"

DEFAULT_TRAIN_PATH = "data/wiki/quality.train"
DEFAULT_VALID_PATH = "data/wiki/quality.valid"

VALID_RATIO = 0.2
VALID_SIZE = 3000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", default=DEFAULT_INPUT_PATH)
    parser.add_argument("--train-path", default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--valid-path", default=DEFAULT_VALID_PATH)
    parser.add_argument("--valid-size", type=int, default=VALID_SIZE)
    parser.add_argument("--valid-ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=40)
    args = parser.parse_args()

    with open(args.input_path) as f:
        lines = f.readlines()

    valid_size = args.valid_ratio if args.valid_ratio else args.valid_size

    with open(args.train_path, "w") as f:
        f.writelines(lines[:-valid_size])

    with open(args.valid_path, "w") as f:
        f.writelines(lines[-valid_size:])

    print(f"Wrote {len(lines) - valid_size} training examples to {args.train_path}")
    print(f"Wrote {valid_size} validation examples to {args.valid_path}")


if __name__ == "__main__":
    main()
