import argparse
import os
import random
from tqdm import tqdm

DATA_DIR = "/data/c-sniderb/a4-leaderboard/classifier/tmp-neg"
OUTPATH = "/data/c-sniderb/a4-leaderboard/classifier/negatives.txt"

NUM_EXAMPLES = 28000


def main(data_dir: str = DATA_DIR, outpath: str = OUTPATH, num_examples: int = NUM_EXAMPLES):
    examples = []

    for file in tqdm(os.listdir(data_dir)):
        with open(os.path.join(data_dir, file)) as f:
            content = f.read()
            if len(content) == 0 or len(content.strip()) == 0:
                continue
            examples.append(content)

    selection = random.sample(examples, num_examples)

    with open(outpath, "w") as f:
        for example in selection:
            joined_text = example.replace("\n", " ")
            f.write(f"__label__negative {joined_text}\n")

    print(f"Wrote {len(selection)} negative examples to {outpath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-examples", type=int, default=NUM_EXAMPLES)
    args = parser.parse_args()
    main(num_examples=args.num_examples)
