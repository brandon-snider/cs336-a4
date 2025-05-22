import os
from cs336_data.exact_deduplication import exact_line_dedupe_docs
from tqdm import tqdm

DATA_DIR = "/data/c-sniderb/a4-leaderboard/classifier/tmp-neg"
OUTDIR = "/data/c-sniderb/a4-leaderboard/classifier/tmp-neg-exact-deduped"


def main(data_dir: str = DATA_DIR, outdir: str = OUTDIR):
    os.makedirs(outdir, exist_ok=True)
    filepaths = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    print(f"Exact deduping {len(filepaths)} files")
    exact_line_dedupe_docs(filepaths, outdir, progress=True)

    # Remove empty files from outdir
    removals = 0
    for f in tqdm(os.listdir(outdir), desc="Removing empty files"):
        path = os.path.join(outdir, f)
        with open(path, "r") as f:
            content = f.read()
            if len(content) == 0 or len(content.strip()) == 0:
                removals += 1
                os.remove(path)

    print(f"Removed {removals} empty files")


if __name__ == "__main__":
    main()
