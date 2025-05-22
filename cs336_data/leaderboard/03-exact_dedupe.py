import argparse
import json
import os
import random
from cs336_data.exact_deduplication import exact_line_dedupe_docs
import time

DATA_DIR = "/data/c-sniderb/a4-leaderboard/02-heuristics"
OUTDIR = "/data/c-sniderb/a4-leaderboard/03-exact-deduped"


def main(data_dir: str = DATA_DIR, outdir: str = OUTDIR, max_files: int = None, mp: bool = False):
    t0 = time.time()
    os.makedirs(outdir, exist_ok=True)
    input_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".warc.wet.gz")]

    if max_files:
        random.seed(42)
        input_paths = random.sample(input_paths, max_files)

    print(f"Deduping {len(input_paths)} files")
    total_lines, unique_lines = exact_line_dedupe_docs(input_paths, outdir, progress=True, mp=mp)
    print(f"Deduped {len(input_paths)} files in {time.time() - t0:.2f} seconds")
    print(f"Total lines: {total_lines:,}")
    print(f"Unique lines: {unique_lines:,}")

    with open(os.path.join(outdir, "meta.json"), "w") as f:
        json.dump(
            {
                "total_files": len(input_paths),
                "total_lines": total_lines,
                "unique_lines": unique_lines,
            },
            f,
            indent=4,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=DATA_DIR)
    parser.add_argument("--outdir", type=str, default=OUTDIR)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--mp", action="store_true")
    args = parser.parse_args()
    main(args.data_dir, args.outdir, args.max_files, args.mp)
