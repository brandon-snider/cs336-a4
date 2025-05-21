import argparse
import random

POS_PATH = "/data/c-sniderb/a4-leaderboard/classifier/positives.txt"
NEG_PATH = "/data/c-sniderb/a4-leaderboard/classifier/negatives.txt"
OUT_PATH = "/data/c-sniderb/a4-leaderboard/classifier/train_all.txt"


def main():
    random.seed(40)

    with open(POS_PATH) as f:
        positive_lines = f.readlines()

    with open(NEG_PATH) as f:
        negative_lines = f.readlines()

    print(f"Read {len(positive_lines)} positive examples")
    print(f"Read {len(negative_lines)} negative examples")

    print(f"Words per positive example: {sum(len(line.split()) for line in positive_lines) / len(positive_lines)}")
    print(f"Words per negative example: {sum(len(line.split()) for line in negative_lines) / len(negative_lines)}")

    lines = positive_lines + negative_lines
    random.shuffle(lines)

    with open(OUT_PATH, "w") as f:
        f.writelines(lines)

    print(f"Wrote {len(lines)} training examples to {OUT_PATH}")


if __name__ == "__main__":
    main()
