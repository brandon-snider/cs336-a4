import json
import os
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


if __name__ == "__main__":
    # main()
    # main2()
    merge_pickles()
