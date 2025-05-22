import random
import os

from tqdm import tqdm

data_dir = "/data/c-sniderb/a4-leaderboard/lang-gopher-exact-deduped"
out_dir = os.path.join(data_dir, "..", "classifier")
tmp_dir = os.path.join(out_dir, "tmp-neg")
out_path = os.path.join(out_dir, "negatives.txt")


def main():
    data_filepaths = [os.path.join(data_dir, filepath) for filepath in os.listdir(data_dir)]
    chosen_paths = random.sample(data_filepaths, 300)
    examples = []

    for data_filepath in tqdm(chosen_paths, desc="Processing files"):
        with open(data_filepath) as f:
            docs = f.read().split("\n\n---END_OF_DOC---\n\n")
            docs = [doc for doc in docs if doc.strip()]

            chosen_docs = random.sample(docs, 60)

            examples.extend(chosen_docs)

    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    for i, example in enumerate(examples):
        tmp_outpath = os.path.join(tmp_dir, f"tmp_neg_{i}.txt")

        with open(tmp_outpath, "w") as f:
            f.write(example)


if __name__ == "__main__":
    main()
