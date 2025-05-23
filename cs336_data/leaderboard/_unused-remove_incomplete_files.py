import argparse
import os


DATA_DIR = "/data/c-sniderb/a4-leaderboard/lang-toxic-gopher"


def main(data_dir: str = DATA_DIR):
    files = set(os.listdir(data_dir))

    # Get all wet files that don't have a meta file
    wet_filepaths = [
        os.path.join(data_dir, f)
        for f in files
        if f.startswith("CC") and f.endswith(".warc.wet.gz") and f + ".meta.json" not in files
    ]

    # Delete the wet files with missing meta files (they're incomplete/corrupted)
    for wet_filepath in wet_filepaths:
        os.remove(wet_filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    args = parser.parse_args()
    main(args.data_dir)
