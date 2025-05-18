import os
from cs336_data.exact_deduplication import exact_line_dedupe
import time

DATA_DIR = "/data/c-sniderb/a4-leaderboard/lang-toxic-gopher"
OUTDIR = "/data/c-sniderb/a4-leaderboard/deduped"


def main(data_dir: str = DATA_DIR, outdir: str = OUTDIR):
    t0 = time.time()

    # Create outdir if it doesn't exist
    os.makedirs(outdir, exist_ok=True)

    files = set(os.listdir(data_dir))

    # Get all wet files that have a meta file
    wet_filepaths = [
        os.path.join(data_dir, f)
        for f in files
        if f.startswith("CC") and f.endswith(".warc.wet.gz") and f + ".meta.json" in files
    ]

    print(f"Deduping {len(wet_filepaths)} files")
    exact_line_dedupe(wet_filepaths, outdir, progress=True)

    print(f"Deduped {len(wet_filepaths)} files in {time.time() - t0} seconds")


if __name__ == "__main__":
    main()
