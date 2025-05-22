import os
from cs336_data.exact_deduplication import exact_line_dedupe_docs
import time

DATA_DIR = "/data/c-sniderb/a4-leaderboard/02-heuristics"
OUTDIR = "/data/c-sniderb/a4-leaderboard/03-exact-deduped"


def main(data_dir: str = DATA_DIR, outdir: str = OUTDIR):
    t0 = time.time()
    os.makedirs(outdir, exist_ok=True)
    files = set(os.listdir(data_dir))

    wet_filepaths = [
        os.path.join(data_dir, f)
        for f in files
        if f.startswith("CC") and f.endswith(".warc.wet.gz") and f + ".meta.json" in files
    ]

    print(f"Deduping {len(wet_filepaths)} files")
    exact_line_dedupe_docs(wet_filepaths, outdir, progress=True)
    print(f"Deduped {len(wet_filepaths)} files in {time.time() - t0:.2f} seconds")


if __name__ == "__main__":
    main()
