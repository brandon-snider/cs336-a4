import argparse
import json
import os

from tqdm import tqdm


DATA_DIR = "/data/c-sniderb/a4-leaderboard/lang-gopher"


def main(data_dir: str = DATA_DIR):
    meta_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".warc.wet.gz.meta.json")]
    merged_meta = {}

    for meta_path in tqdm(meta_paths):
        filename = os.path.basename(meta_path).replace(".meta.json", "")

        with open(meta_path) as f:
            meta = json.load(f)

        merged_meta[filename] = {k: v for k, v in meta.items() if k not in ["accepted_docs", "rejected_docs"]}

    with open(os.path.join(data_dir, "merged_meta.json"), "w") as f:
        json.dump(merged_meta, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    args = parser.parse_args()

    main(args.data_dir)
