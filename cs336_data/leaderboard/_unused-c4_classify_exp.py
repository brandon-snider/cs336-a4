import os
import random
from cs336_data.leaderboard.classifier.c4_100_classifier import classify_c4_100
from transformers import AutoTokenizer

from tqdm import tqdm

DATA_DIR = "/data/c-sniderb/a4-leaderboard/lang-gopher"
OUT_DIR = "/data/c-sniderb/a4-leaderboard/classified"

MAX_FILES = 10

TOKENIZER = AutoTokenizer.from_pretrained("gpt2")


def main(data_dir: str = DATA_DIR, out_dir: str = OUT_DIR, max_files: int = MAX_FILES):
    os.makedirs(out_dir, exist_ok=True)

    filepaths = sorted(
        [os.path.join(data_dir, filepath) for filepath in os.listdir(data_dir) if filepath.endswith(".warc.wet.gz")]
    )

    # random.seed(42)
    random.shuffle(filepaths)
    positives = {}

    # Min confidence -> n_repeats
    # brackets = {
    #     0.9: 6,
    #     0.75: 4,
    #     0.5: 3,
    #     0.3: 2,
    #     0.1: 1,
    # }

    brackets = {0.005: 1}

    for min_conf, n_repeats in brackets.items():
        positives[min_conf] = {
            "docs_count": 0,
            "unique_docs_count": 0,
            "tokens_count": 0,
            "unique_tokens_count": 0,
        }

    negative_examples = []

    for filepath in tqdm(filepaths[:max_files], desc="Files"):
        with open(filepath) as f:
            docs = f.read().split("\n\n---END_OF_DOC---\n\n")
            docs = [doc for doc in docs if doc.strip()]

            # for doc in tqdm(docs, desc="Docs in file"):
            for doc in docs:
                label, conf = classify_c4_100(doc)
                pos_score = conf if label == "positive" else 1 - conf

                if pos_score < 0.1:
                    negative_examples.append(doc)
                    if len(negative_examples) >= 30:
                        break
                    continue

                for min_conf, n_repeats in brackets.items():
                    if pos_score > min_conf:
                        token_count = len(TOKENIZER.encode(doc))

                        positives[min_conf]["unique_docs_count"] += 1
                        positives[min_conf]["unique_tokens_count"] += token_count

                        positives[min_conf]["docs_count"] += n_repeats
                        positives[min_conf]["tokens_count"] += n_repeats * token_count

                        break

        if len(negative_examples) >= 30:
            break

    print(f"Collected {len(negative_examples)} negative examples")

    for negative_example in negative_examples:
        print(negative_example)
        print("-" * 100)

    for min_conf, stats in positives.items():
        print(
            f"Conf > {min_conf}: {stats['docs_count']:,} docs ({stats['unique_docs_count']:,} unique) | {stats['tokens_count']:,} tokens ({stats['unique_tokens_count']:,} unique)"
        )

    print("-" * 100)

    #     if len(positive["doc"]) > 200:
    #         start = random.randint(0, len(positive["doc"]) - 200)
    #         print(positive["doc"][start : start + 200])
    #     else:
    #         print(positive)

    tokens_per_file = sum([stats["tokens_count"] for stats in positives.values()]) / max_files
    print("-" * 100)
    print(f"Tokens per file: {tokens_per_file:,}")
    print(f"Est. total tokens: {tokens_per_file * 5000:,}")


if __name__ == "__main__":
    main()
