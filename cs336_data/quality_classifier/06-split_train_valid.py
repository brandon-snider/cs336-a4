import argparse

DEFAULT_INPUT_PATH = "data/wiki/train_all.txt"
DEFAULT_TRAIN_PATH = "data/wiki/quality.train"
DEFAULT_VALID_PATH = "data/wiki/quality.valid"

VALID_RATIO = 0.1
# VALID_SIZE = 3000


def main(
    input_path: str,
    train_path: str,
    valid_path: str,
    valid_size: int | None,
    valid_ratio: float,
):
    with open(input_path) as f:
        lines = f.readlines()

    valid_size = valid_size if valid_size else int(len(lines) * valid_ratio)

    with open(train_path, "w") as f:
        f.writelines(lines[:-valid_size])

    with open(valid_path, "w") as f:
        f.writelines(lines[-valid_size:])

    print(f"Wrote {len(lines) - valid_size} training examples to {train_path}")
    print(f"Wrote {valid_size} validation examples to {valid_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", default=DEFAULT_INPUT_PATH)
    parser.add_argument("--train-path", default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--valid-path", default=DEFAULT_VALID_PATH)
    parser.add_argument("--valid-size", type=int, default=None)
    parser.add_argument("--valid-ratio", type=float, default=VALID_RATIO)
    args = parser.parse_args()

    main(args.input_path, args.train_path, args.valid_path, args.valid_size, args.valid_ratio, args.seed)
