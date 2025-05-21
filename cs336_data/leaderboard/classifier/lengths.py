import argparse
import os
import random
import matplotlib.pyplot as plt

POS_PATH = "/data/c-sniderb/a4-leaderboard/classifier/positives.txt"
NEG_PATH = "/data/c-sniderb/a4-leaderboard/classifier/negatives.txt"


def main():
    random.seed(40)

    with open(POS_PATH) as f:
        positive_lines = f.readlines()

    with open(NEG_PATH) as f:
        negative_lines = f.readlines()

    pos_lengths = [len(line.split()) for line in positive_lines]
    neg_lengths = [len(line.split()) for line in negative_lines]

    if not os.path.exists("out/plots"):
        os.makedirs("out/plots")

    # Positive plot
    plt.figure(figsize=(10, 5))
    plt.hist(pos_lengths, bins=30, alpha=0.7, color="blue")
    plt.xlabel("Number of Words")
    plt.ylabel("Count")
    plt.title("Histogram of Positive Example Lengths")
    plt.tight_layout()
    plt.savefig("out/plots/lengths_positive.png")
    plt.show()

    # Negative plot
    plt.figure(figsize=(10, 5))
    plt.hist(neg_lengths, bins=30, alpha=0.7, color="red")
    plt.xlabel("Number of Words")
    plt.ylabel("Count")
    plt.title("Histogram of Negative Example Lengths")
    plt.tight_layout()
    plt.savefig("out/plots/lengths_negative.png")
    plt.show()


if __name__ == "__main__":
    main()
