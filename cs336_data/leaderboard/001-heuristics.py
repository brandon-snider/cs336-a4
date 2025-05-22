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

from cs336_data.c4_quality_filters import c4_quality_filter
from cs336_data.gopher_quality_filters import gopher_quality_filter

# from transformers import AutoTokenizer
# TOKENIZER = AutoTokenizer.from_pretrained("gpt2")

DATA_DIR = "/data/c-sniderb/a4-leaderboard/01-english"
OUTDIR = "/data/c-sniderb/a4-leaderboard/02-heuristics"


def process_file(input_path: str, output_path: str, progress: bool = False):
    accepted_docs_count = 0
    rejected_docs = {"blacklisted": 0, "no_lines_kept": 0, "gopher": 0, "nsfw": 0, "toxic": 0}

    accepted_lines_count = 0

    rejected_lines = {
        "short": 0,
        "blacklisted": 0,
        "invalid_terminator": 0,
    }

    tokens = {"total": 0, "kept": 0, "rejected": 0}

    with open(output_path, "w") as fout:
        with open(input_path) as fin:
            t0 = time.time()

            docs = fin.read().split("\n\n---END_OF_DOC---\n\n")

            for doc in tqdm(docs, total=len(docs), desc="Processing documents", disable=not progress):
                rejected_docs_count = sum(rejected_docs.values())
                total_docs = accepted_docs_count + rejected_docs_count

                if total_docs % 1000 == 0 and progress:
                    t1 = time.time()
                    time_per_doc_ms = (t1 - t0) / total_docs * 1000

                    rejected_lines_count = sum(rejected_lines.values())
                    total_lines = accepted_lines_count + rejected_lines_count

                    docs_kept_pct = accepted_docs_count / total_docs
                    docs_rejected_pct = rejected_docs_count / total_docs
                    lines_kept_pct = accepted_lines_count / total_lines
                    lines_rejected_pct = rejected_lines_count / total_lines

                    print(
                        f"{total_docs:,} docs | Kept {accepted_docs_count:,} ({docs_kept_pct:.2%}) | Rejected {rejected_docs_count:,} ({docs_rejected_pct:.2%}) | {accepted_lines_count:,} accepted lines ({lines_kept_pct:.2%}) | {rejected_lines_count:,} rejected lines ({lines_rejected_pct:.2%}) | {time_per_doc_ms:.2f}ms/doc",
                        end="\r",
                    )

                is_c4_quality, filtered_doc, metadata = c4_quality_filter(doc)

                # tokens["total"] += len(TOKENIZER.encode(doc))

                if not is_c4_quality:
                    if metadata["reason"] == "blacklisted":
                        rejected_docs["blacklisted"] += 1
                    elif metadata["reason"] == "no_lines_kept":
                        rejected_docs["no_lines_kept"] += 1
                    continue

                is_gopher_quality = gopher_quality_filter(filtered_doc)
                if not is_gopher_quality:
                    rejected_docs["gopher"] += 1
                    continue

                # nsfw_label, nsfw_conf = classify_nsfw(text)
                # non_nsfw_conf = nsfw_conf if nsfw_label == "non-nsfw" else 1 - nsfw_conf
                # if non_nsfw_conf < 0.9:
                #     rejected_docs["nsfw"] += 1
                #     continue

                # toxic_label, toxic_conf = classify_toxic_speech(text)
                # non_toxic_conf = toxic_conf if toxic_label == "non-toxic" else 1 - toxic_conf
                # if non_toxic_conf < 0.8:
                #     rejected_docs["toxic"] += 1
                #     continue

                # tokens["kept"] += len(TOKENIZER.encode(filtered_doc))
                # tokens["rejected"] = tokens["total"] - tokens["kept"]

                accepted_docs_count += 1
                accepted_lines_count += metadata["line_meta"]["kept"]

                rejected_lines["short"] += metadata["line_meta"]["short"]
                rejected_lines["blacklisted"] += metadata["line_meta"]["blacklisted"]
                rejected_lines["invalid_terminator"] += metadata["line_meta"]["invalid_terminator"]

                fout.write(filtered_doc + "\n\n---END_OF_DOC---\n\n")

    meta = {
        "total_docs": total_docs,
        "accepted_docs_ct": accepted_docs_count,
        "rejected_docs_ct": sum(rejected_docs.values()),
        "accepted_lines_in_accepted_docs_ct": accepted_lines_count,
        "rejected_lines_in_accepted_docs_ct": sum(rejected_lines.values()),
        "rejected_docs_by_type": {
            "blacklisted": rejected_docs["blacklisted"],
            "no_lines_kept": rejected_docs["no_lines_kept"],
            "gopher": rejected_docs["gopher"],
        },
        "rejected_lines_in_accepted_docs_by_type": {
            "short": rejected_lines["short"],
            "blacklisted": rejected_lines["blacklisted"],
            "invalid_terminator": rejected_lines["invalid_terminator"],
        },
        "tokens": tokens,
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
