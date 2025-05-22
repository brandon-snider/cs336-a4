import json
import os
import pathlib
import pickle


def main():
    data_dir = "/data/c-sniderb/a4-leaderboard/lang-toxic-gopher"
    filepaths = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.startswith("CC") and f.endswith(".warc.wet.gz.meta.json")
    ]

    stats = {
        "total_docs": 0,
        "total_accepted_docs": 0,
        "total_rejected_docs": 0,
        "total_rejected_language": 0,
        "total_rejected_nsfw": 0,
        "total_rejected_toxic": 0,
        "total_rejected_gopher": 0,
    }

    for filepath in filepaths:
        with open(filepath) as f:
            meta = json.load(f)

            stats["total_docs"] += meta["total_docs"]
            stats["total_accepted_docs"] += meta["accepted_ct"]
            stats["total_rejected_docs"] += meta["rejected_ct"]
            stats["total_rejected_language"] += meta["rejected_by_type"]["language"]
            stats["total_rejected_nsfw"] += meta["rejected_by_type"]["nsfw"]
            stats["total_rejected_toxic"] += meta["rejected_by_type"]["toxic"]
            stats["total_rejected_gopher"] += meta["rejected_by_type"]["gopher_quality"]

    print(stats)


def main2():
    data_dir = "/data/c-sniderb/a4-leaderboard/deduped"
    filepaths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".warc.wet.gz")]

    filepath = filepaths[0]

    with open(filepath) as f:
        text = f.read()
        print(text[:1000])


def merge_pickles():
    data_dir = "/data/c-sniderb/a4-leaderboard/near-deduped"
    filepaths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pkl")]

    signatures = []

    for filepath in filepaths:
        with open(filepath, "rb") as f:
            signatures.extend(pickle.load(f))

    outpath = os.path.join(data_dir, "signatures.pkl")

    with open(outpath, "wb") as f:
        pickle.dump(signatures, f)

    print(f"Dumped {len(signatures)} signatures to {outpath}")


def clear_failed_reservations():
    submission_dir = "/data/c-sniderb/a4-leaderboard/slurm_logs"
    outdir = "/data/c-sniderb/a4-leaderboard/lang-toxic-gopher"
    for filepath in os.listdir(submission_dir):
        if filepath.endswith(".pkl") and not filepath.startswith("2"):
            with open(os.path.join(submission_dir, filepath), "rb") as f:
                result = pickle.load(f)
                wet_filepath = result.args[0]
                reservation_path = os.path.join(outdir, str(pathlib.Path(wet_filepath).name)) + ".reservation.txt"

                if os.path.exists(reservation_path):
                    os.remove(reservation_path)


def get_filter_stats():
    data_dir = "/data/c-sniderb/a4-leaderboard/lang-gopher"
    merged_meta_path = os.path.join(data_dir, "merged_meta.json")

    with open(merged_meta_path) as f:
        merged_meta = json.load(f)

    accepted_docs = 0
    rejected_docs = 0
    total_docs = 0

    rejected_language = 0
    rejected_gopher = 0

    for filename, meta in merged_meta.items():
        accepted_docs += meta["accepted_ct"]
        rejected_docs += meta["rejected_ct"]
        total_docs += meta["total_docs"]

        rejected_language += meta["rejected_by_type"]["language"]
        rejected_gopher += meta["rejected_by_type"]["gopher_quality"]

    print(f"Total docs: {total_docs:,}")
    print(f"Accepted docs: {accepted_docs:,}")
    print(f"Rejected docs: {rejected_docs:,}")
    print(f"Acceptance rate: {accepted_docs / total_docs:.2%}")
    print("-" * 100)
    print(f"Rejected language: {rejected_language:,} ({rejected_language / total_docs:.2%})")

    rejected_gopher_after_language = rejected_gopher / (total_docs - rejected_language)

    print(
        f"Rejected gopher: {rejected_gopher:,} ({rejected_gopher / total_docs:.2%}) | {rejected_gopher_after_language:.2%} after language"
    )


if __name__ == "__main__":
    # main()
    # main2()
    # merge_pickles()
    # clear_failed_reservations()
    pass
