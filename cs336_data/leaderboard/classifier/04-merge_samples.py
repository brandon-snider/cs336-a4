import random

POS_TRAIN_PATH = "/data/c-sniderb/a4-leaderboard/classifier/positives_train.txt"
POS_VALID_PATH = "/data/c-sniderb/a4-leaderboard/classifier/positives_valid.txt"
NEG_TRAIN_PATH = "/data/c-sniderb/a4-leaderboard/classifier/negatives_train.txt"
NEG_VALID_PATH = "/data/c-sniderb/a4-leaderboard/classifier/negatives_valid.txt"
TRAIN_OUT_PATH = "/data/c-sniderb/a4-leaderboard/classifier/quality.train"
VALID_OUT_PATH = "/data/c-sniderb/a4-leaderboard/classifier/quality.valid"


def main():
    random.seed(42)

    with open(POS_TRAIN_PATH) as f:
        positive_lines = f.readlines()

    with open(NEG_TRAIN_PATH) as f:
        negative_lines = f.readlines()

    print(f"Read {len(positive_lines)} positive examples")
    print(f"Read {len(negative_lines)} negative examples")

    print(f"Words per positive example: {sum(len(line.split()) for line in positive_lines) / len(positive_lines)}")
    print(f"Words per negative example: {sum(len(line.split()) for line in negative_lines) / len(negative_lines)}")

    lines = positive_lines + negative_lines
    random.shuffle(lines)

    with open(TRAIN_OUT_PATH, "w") as f:
        f.writelines(lines)

    print(f"Wrote {len(lines)} training examples to {TRAIN_OUT_PATH}")

    with open(POS_VALID_PATH) as f:
        positive_valid_lines = f.readlines()

    with open(NEG_VALID_PATH) as f:
        negative_valid_lines = f.readlines()

    lines = positive_valid_lines + negative_valid_lines
    random.shuffle(lines)

    with open(VALID_OUT_PATH, "w") as f:
        f.writelines(lines)

    print(f"Wrote {len(lines)} validation examples to {VALID_OUT_PATH}")


if __name__ == "__main__":
    main()
