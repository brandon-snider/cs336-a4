import os
import random
from cs336_data.gopher_quality_filters import gopher_quality_filter
from cs336_data.leaderboard.classifier.c4_100_classifier import classify_c4_100
from transformers import AutoTokenizer

from tqdm import tqdm

DATA_DIR = "/data/c-sniderb/a4-leaderboard/lang-gopher"
OUT_DIR = "/data/c-sniderb/a4-leaderboard/classified"

MAX_FILES = 5

TOKENIZER = AutoTokenizer.from_pretrained("gpt2")


def main(data_dir: str = DATA_DIR, out_dir: str = OUT_DIR, max_files: int = MAX_FILES):
    os.makedirs(out_dir, exist_ok=True)

    filepaths = sorted(
        [os.path.join(data_dir, filepath) for filepath in os.listdir(data_dir) if filepath.endswith(".warc.wet.gz")]
    )

    # random.seed(42)
    random.shuffle(filepaths)

    positives = []

    for filepath in tqdm(filepaths[:max_files], desc="Files"):
        with open(filepath) as f:
            docs = f.read().split("\n\n---END_OF_DOC---\n\n")
            docs = [doc for doc in docs if doc.strip()]

            # for doc in tqdm(docs, desc="Docs in file"):
            for doc in docs:
                label, conf = classify_c4_100(doc)
                pos_score = conf if label == "positive" else 1 - conf

                # n_repeats = (
                #     6
                #     if pos_score > 0.95
                #     else 4
                #     if pos_score > 0.8
                #     else 3
                #     if pos_score > 0.5
                #     else 2
                #     if pos_score > 0.3
                #     else 1
                #     if pos_score > 0.1
                #     else 0
                # )

                # if n_repeats == 0:
                #     continue

                n_repeats = 1

                if pos_score < 0.9:
                    continue

                token_count = len(TOKENIZER.encode(doc))

                for _ in range(n_repeats):
                    positives.append(
                        {
                            "doc": doc,
                            "conf": pos_score,
                            "token_count": token_count,
                        }
                    )

    for positive in positives:
        print(f"Token count: {positive['token_count']} | Confidence: {positive['conf']}")

        if len(positive["doc"]) > 200:
            start = random.randint(0, len(positive["doc"]) - 200)
            print(positive["doc"][start : start + 200])
        else:
            print(positive)

    tokens_per_file = sum([positive["token_count"] for positive in positives]) / max_files
    print("-" * 100)
    print(f"Tokens per file: {tokens_per_file:,}")
    print(f"Est. total tokens: {tokens_per_file * 5000:,}")


if __name__ == "__main__":
    main()
