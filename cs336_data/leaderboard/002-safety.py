"""
Filter pages and lines using the C4 and Gopher quality heuristics.
"""

import argparse
import json
import os
import random
import time
from tqdm import tqdm
import pathlib
import submitit
import concurrent.futures

from cs336_data.harmful_content import classify_nsfw, classify_toxic_speech

# DATA_DIR = "/data/c-sniderb/a4-leaderboard/02-heuristics"
DATA_DIR = "/data/c-sniderb/a4-leaderboard/heuristics"
OUTDIR = "/data/c-sniderb/a4-leaderboard/03-safety"


def process_file(input_path: str, output_path: str, progress: bool = False):
    accepted_docs_count = 0
    rejected_docs = {"nsfw": 0, "toxic": 0}

    with open(output_path, "w") as fout:
        with open(input_path) as fin:
            docs = fin.read().split("\n\n---END_OF_DOC---\n\n")

            for doc in tqdm(docs, total=len(docs), desc="Processing documents", disable=not progress):
                nsfw_label, nsfw_conf = classify_nsfw(doc)
                non_nsfw_conf = nsfw_conf if nsfw_label == "non-nsfw" else 1 - nsfw_conf
                if non_nsfw_conf < 0.9:
                    rejected_docs["nsfw"] += 1
                    continue

                toxic_label, toxic_conf = classify_toxic_speech(doc)
                non_toxic_conf = toxic_conf if toxic_label == "non-toxic" else 1 - toxic_conf
                if non_toxic_conf < 0.8:
                    rejected_docs["toxic"] += 1
                    continue

                accepted_docs_count += 1

                fout.write(doc + "\n\n---END_OF_DOC---\n\n")

    rejected_docs_count = sum(rejected_docs.values())
    total_docs = accepted_docs_count + rejected_docs_count

    meta = {
        "total_docs": total_docs,
        "accepted_docs_ct": accepted_docs_count,
        "rejected_docs_ct": rejected_docs_count,
        "rejected_docs_by_type": {
            "nsfw": rejected_docs["nsfw"],
            "toxic": rejected_docs["toxic"],
        },
    }

    meta_outpath = f"{output_path}.meta.json"
    with open(meta_outpath, "w") as f:
        json.dump(meta, f, indent=4)

    return output_path, meta_outpath, meta


def main(
    max_files: int = None,
    single: bool = False,
    outdir: str = OUTDIR,
    data_dir: str = DATA_DIR,
    wait: bool = True,
    mp: bool = False,
):
    random.seed(42)

    os.makedirs(outdir, exist_ok=True)

    all_wet_filepaths = [
        os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.startswith("CC") and f.endswith(".warc.wet.gz")
    ]

    wet_filepaths = []
    for wfp in all_wet_filepaths:
        outpath = os.path.join(outdir, str(pathlib.Path(wfp).name))
        reservation_path = outpath + ".reservation.txt"
        if not os.path.exists(outpath) and not os.path.exists(reservation_path):
            wet_filepaths.append(wfp)

    if max_files is not None:
        random.shuffle(wet_filepaths)
        wet_filepaths = wet_filepaths[:max_files]

    for wfp in wet_filepaths:
        reservation_path = os.path.join(outdir, str(pathlib.Path(wfp).name)) + ".reservation.txt"
        if not os.path.exists(reservation_path):
            with open(reservation_path, "w") as f:
                f.write("1")

    if single:
        for wfp in tqdm(wet_filepaths, total=len(wet_filepaths)):
            output_file, meta_outpath, meta = process_file(
                wfp, os.path.join(outdir, str(pathlib.Path(wfp).name)), progress=True
            )
            print(f"Output file written: {output_file}")
            print(f"Meta file written: {meta_outpath}")
            print(f"Meta: {meta}")
    elif mp:
        num_cpus = len(os.sched_getaffinity(0))
        print(f"Using {num_cpus} CPUs")
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=num_cpus)

        futures = []

        for wfp in wet_filepaths:
            wet_filename = str(pathlib.Path(wfp).name)
            future = executor.submit(process_file, wfp, os.path.join(outdir, wet_filename), progress=False)
            futures.append(future)

        if wait:
            for future in tqdm(futures, total=len(wet_filepaths)):
                output_file, meta_outpath, meta = future.result()

    else:
        executor = submitit.AutoExecutor(folder="/data/c-sniderb/a4-leaderboard/slurm_logs")
        max_simultaneous_jobs = 64

        executor.update_parameters(
            slurm_array_parallelism=max_simultaneous_jobs,
            timeout_min=10,
            mem_gb=2,
            cpus_per_task=1,
            slurm_account="student",
            slurm_partition="a4-cpu",
            slurm_qos="a4-cpu-qos",
        )

        futures = []

        with executor.batch():
            for wfp in wet_filepaths:
                wet_filename = str(pathlib.Path(wfp).name)
                future = executor.submit(process_file, wfp, os.path.join(outdir, wet_filename))
                futures.append(future)

        if wait:
            # Iterate over completed futures as they finish, using a progress bar to keep track
            for future in tqdm(
                submitit.helpers.as_completed(futures),
                total=len(wet_filepaths),
            ):
                output_file, meta_outpath, short_meta = future.result()
                # print(f"Output file written: {output_file}")
                # print(f"Meta file written: {meta_outpath}")
                # print(f"Short Meta: {short_meta}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-files", type=int, default=None, help="Maximum number of files to process")
    parser.add_argument("--single", action="store_true", help="Whether to use a single thread")
    parser.add_argument("--out-dir", type=str, default=OUTDIR, help="Output directory")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR, help="Data directory")
    parser.add_argument("--no-wait", action="store_true", help="Whether to wait for the jobs to finish")
    parser.add_argument("--mp", action="store_true", help="Whether to use multiple processes")
    args = parser.parse_args()

    wait = not args.no_wait

    main(
        max_files=args.max_files,
        single=args.single,
        outdir=args.out_dir,
        data_dir=args.data_dir,
        wait=wait,
        mp=args.mp,
    )
