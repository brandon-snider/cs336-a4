import os
from tqdm import tqdm
from cs336_data.minhash_deduplication_serial import minhash_dedupe

DATA_DIR = "/data/c-sniderb/a4-leaderboard/classifier/tmp-neg-exact-deduped"
OUTDIR = "/data/c-sniderb/a4-leaderboard/classifier/tmp-neg-near-deduped"


def main(data_dir: str = DATA_DIR, outdir: str = OUTDIR):
    os.makedirs(outdir, exist_ok=True)
    filepaths = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    print(f"Near deduping {len(filepaths)} files")

    minhash_dedupe(
        input_files=filepaths,
        output_directory=outdir,
        num_hashes=100,
        num_bands=10,
        ngrams=5,
        jaccard_threshold=0.8,
        progress=True,
    )

    # Remove empty files from outdir
    removals = 0
    for f in tqdm(os.listdir(outdir), desc="Removing empty files"):
        path = os.path.join(outdir, f)
        try:
            with open(path) as f:
                content = f.read()
                if len(content) == 0 or len(content.strip()) == 0:
                    removals += 1
                    os.remove(path)
        except Exception as e:
            print(f"Error reading {path}: {e}")
            removals += 1
            os.remove(path)

    print(f"Removed {removals} empty files")


if __name__ == "__main__":
    main()
