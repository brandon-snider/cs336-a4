import argparse
import os
import pathlib
import pickle

SUBMISSION_DIR = "/data/c-sniderb/a4-leaderboard/slurm_logs"
OUTDIR = "/data/c-sniderb/a4-leaderboard/lang-toxic-gopher"


def main(submission_dir: str = SUBMISSION_DIR, outdir: str = OUTDIR):
    for filepath in os.listdir(submission_dir):
        if filepath.endswith(".pkl") and not filepath.endswith("_submitted.pkl"):
            with open(os.path.join(submission_dir, filepath), "rb") as f:
                result = pickle.load(f)
                wet_filepath = result.args[0]
                reservation_path = os.path.join(outdir, str(pathlib.Path(wet_filepath).name)) + ".reservation.txt"

                if os.path.exists(reservation_path):
                    os.remove(reservation_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission_dir", type=str, default=SUBMISSION_DIR)
    parser.add_argument("--outdir", type=str, default=OUTDIR)
    args = parser.parse_args()

    main(args.submission_dir, args.outdir)
