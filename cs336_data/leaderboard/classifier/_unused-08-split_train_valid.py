import argparse

INPUT_PATH = "/data/c-sniderb/a4-leaderboard/classifier/train_all.txt"
TRAIN_PATH = "/data/c-sniderb/a4-leaderboard/classifier/quality.train"
VALID_PATH = "/data/c-sniderb/a4-leaderboard/classifier/quality.valid"

VALID_RATIO = 0.1
# VALID_SIZE = 3000


def main():
    with open(INPUT_PATH) as f:
        lines = f.readlines()

    valid_size = int(len(lines) * VALID_RATIO)

    with open(TRAIN_PATH, "w") as f:
        f.writelines(lines[:-valid_size])

    with open(VALID_PATH, "w") as f:
        f.writelines(lines[-valid_size:])

    print(f"Wrote {len(lines) - valid_size} training examples to {TRAIN_PATH}")
    print(f"Wrote {valid_size} validation examples to {VALID_PATH}")


if __name__ == "__main__":
    main()
