import argparse
import concurrent.futures
import json
import os
import random

import submitit
from cs336_data.leaderboard.classifier.c4_100_classifier import classify_c4_100
from tqdm import tqdm
from transformers import AutoTokenizer

TOKENIZER = AutoTokenizer.from_pretrained("gpt2")

DATA_DIR = "/data/c-sniderb/a4-leaderboard/03-exact-deduped"
OUT_DIR = "/data/c-sniderb/a4-leaderboard/04-classified-bucketed"

MAX_FILES = 10_000
CHUNK_SIZE = 30


def deep_add_merge(a, b):
    for k, v in b.items():
        if k in a:
            if isinstance(v, dict) and isinstance(a[k], dict):
                deep_add_merge(a[k], v)
            elif isinstance(v, int | float) and isinstance(a[k], int | float):
                a[k] += v
            else:
                a[k] = v
        else:
            a[k] = v

    return a


def process_file(inpath: str, outpath: str, brackets: dict[float, int]):
    stats = {}
    for min_conf, n_repeats in brackets.items():
        stats[min_conf] = {
            "unique_docs_count": 0,
            "docs_count": 0,
            "tokens_count": 0,
            "unique_tokens_count": 0,
        }

    with open(inpath) as fin, open(outpath, "w") as fout:
        docs = fin.read().split("\n\n---END_OF_DOC---\n\n")

        for doc in docs:
            if not doc.strip():
                continue

            label, conf = classify_c4_100(doc)
            pos_score = conf if label == "positive" else 1 - conf

            min_conf = max((th for th in brackets if pos_score > th), default=0)
            n_repeats = brackets[min_conf]

            stats[min_conf]["unique_docs_count"] += 1
            stats[min_conf]["docs_count"] += n_repeats

            # token_count = len(TOKENIZER.encode(doc))
            # stats[min_conf]["unique_tokens_count"] += token_count if n_repeats > 0 else 0
            # stats[min_conf]["tokens_count"] += n_repeats * token_count

            for _ in range(n_repeats):
                fout.write(doc + "\n\n---END_OF_DOC---\n\n")

    return stats


def process_file_chunk(filepaths: list[str], out_dir: str, brackets: dict[float, int]):
    stats_list = []

    for filepath in filepaths:
        outpath = os.path.join(out_dir, os.path.basename(filepath))
        stats_list.append(process_file(filepath, outpath, brackets))

    return stats_list


def main(
    data_dir: str = DATA_DIR,
    out_dir: str = OUT_DIR,
    max_files: int = None,
    chunk_size: int = CHUNK_SIZE,
    single: bool = False,
    mp: bool = False,
    threshold: float = None,
):
    os.makedirs(out_dir, exist_ok=True)

    filepaths = sorted(
        [os.path.join(data_dir, filepath) for filepath in os.listdir(data_dir) if filepath.endswith(".warc.wet.gz")]
    )

    random.seed(42)
    random.shuffle(filepaths)

    if max_files is not None:
        filepaths = filepaths[:max_files]

    is_bracketed = threshold is None
    if is_bracketed:
        # Min confidence -> n_repeats
        brackets = {0.84: 4, 0.72: 3, 0.58: 2, 0.36: 1, 0.0: 0}
    else:
        brackets = {threshold: 1, 0.0: 0}

    stats_list = []

    if single:
        for filepath in tqdm(filepaths, desc="Files"):
            outpath = os.path.join(out_dir, os.path.basename(filepath))
            stats_list.append(process_file(filepath, outpath, brackets))
    elif mp:
        num_cpus = len(os.sched_getaffinity(0))
        print(f"Using {num_cpus} CPUs")
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=num_cpus)

        futures = []

        for filepath in filepaths:
            future = executor.submit(
                process_file, filepath, os.path.join(out_dir, os.path.basename(filepath)), brackets
            )
            futures.append(future)

        for future in tqdm(futures, total=len(futures)):
            out = future.result()
            stats_list.append(out)
    else:
        executor = submitit.AutoExecutor(folder="/data/c-sniderb/a4-leaderboard/slurm_logs")
        max_simultaneous_jobs = 16
        executor.update_parameters(
            slurm_array_parallelism=max_simultaneous_jobs,
            timeout_min=10,
            mem_gb=4,
            cpus_per_task=1,
            slurm_account="student",
            slurm_partition="a4-cpu",
            slurm_qos="a4-cpu-qos",
        )

        futures = []

        with executor.batch():
            for i in range(0, len(filepaths), chunk_size):
                chunk = filepaths[i : i + chunk_size]
                future = executor.submit(process_file_chunk, chunk, out_dir, brackets)
                futures.append(future)

        for future in tqdm(submitit.helpers.as_completed(futures), total=len(futures)):
            out = future.result()
            stats_list.extend(out)

    stats = {}
    for s in stats_list:
        stats = deep_add_merge(stats, s)

    total_tokens = sum(stats[th]["tokens_count"] for th in stats)
    total_unique_tokens = sum(stats[th]["unique_tokens_count"] for th in stats)

    stats["total"] = {
        "tokens_count": total_tokens,
        "unique_tokens_count": total_unique_tokens,
        "est_total_tokens": total_tokens / len(filepaths) * 5000,
        "est_total_unique_tokens": total_unique_tokens / len(filepaths) * 5000,
    }

    with open(os.path.join(out_dir, "stats.json"), "w") as fin:
        json.dump(stats, fin, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-files", type=int, default=MAX_FILES)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--data-dir", type=str, default=DATA_DIR)
    parser.add_argument("--out-dir", type=str, default=OUT_DIR)
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--thresholded", action="store_true")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--mp", action="store_true")
    args = parser.parse_args()

    if args.thresholded:
        threshold = args.threshold or 0.1

    main(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        max_files=args.max_files,
        chunk_size=args.chunk_size,
        single=args.single,
        mp=args.mp,
        threshold=args.threshold,
    )
