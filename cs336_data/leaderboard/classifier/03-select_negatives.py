import random
import os

from tqdm import tqdm

data_dir = "/data/c-sniderb/a4-leaderboard/lang-gopher"
out_dir = os.path.join(data_dir, "..", "classifier")
tmp_dir = os.path.join(out_dir, "tmp-neg")
out_path = os.path.join(out_dir, "negatives.txt")


def main():
    all_filepaths = os.listdir(data_dir)
    meta_filepaths = set([filepath for filepath in all_filepaths if filepath.endswith(".meta.json")])

    data_filepaths = [
        os.path.join(data_dir, filepath)
        for filepath in all_filepaths
        if filepath.endswith(".warc.wet.gz") and f"{filepath}.meta.json" in meta_filepaths
    ]

    examples = []

    chosen_paths = random.sample(data_filepaths, 300)

    for data_filepath in tqdm(chosen_paths, desc="Processing files"):
        with open(data_filepath) as f:
            docs = f.read().split("\n\n---END_OF_DOC---\n\n")
            docs = [doc for doc in docs if doc.strip()]

            chosen_docs = random.sample(docs, 40)

            examples.extend(chosen_docs)

    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    for i, example in enumerate(examples):
        tmp_outpath = os.path.join(tmp_dir, f"tmp_neg_{i}.txt")

        with open(tmp_outpath, "w") as f:
            f.write(example)


if __name__ == "__main__":
    main()
