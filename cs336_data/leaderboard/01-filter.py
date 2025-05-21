import argparse
import json
import os
import random
import time
from tqdm import tqdm
import pathlib
import submitit

from fastwarc.warc import ArchiveIterator
from cs336_data.extract_text import extract_text_from_html_bytes
from cs336_data.gopher_quality_filters import gopher_quality_filter
from cs336_data.harmful_content import classify_nsfw, classify_toxic_speech
from cs336_data.language_identification import identify_language

OUTDIR = "/data/c-sniderb/a4-leaderboard/lang-gopher"
DATA_DIR = "/data/CC"


def process_wet_file(input_path: str, output_path: str, progress: bool = False):
    total_docs = 0
    accepted_docs = []

    rejected_docs = {
        "language": [],
        "nsfw": [],
        "toxic": [],
        "gopher_quality": [],
    }

    with open(output_path, "w") as accepted_file:
        t0 = time.time()
        for record in ArchiveIterator(open(input_path, "rb")):
            total_docs += 1
            if total_docs % 1000 == 0 and progress:
                t1 = time.time()
                time_per_doc_ms = (t1 - t0) / total_docs * 1000
                print(
                    f"Processed {total_docs} records | {len(accepted_docs)} accepted | {len(rejected_docs['language'])} rejected (language) | {len(rejected_docs['nsfw'])} rejected (nsfw) | {len(rejected_docs['toxic'])} rejected (toxic) | {len(rejected_docs['gopher_quality'])} rejected (gopher_quality) | {time_per_doc_ms:.2f}ms/doc",
                    end="\r",
                )

            text = extract_text_from_html_bytes(record.reader.read())

            lang, score = identify_language(text)

            if lang != "en" or score < 0.85:
                rejected_docs["language"].append([record.record_id, lang, score])
                continue

            # nsfw_label, nsfw_conf = classify_nsfw(text)
            # if nsfw_label == "nsfw" or (nsfw_label == "non-nsfw" and nsfw_conf < 0.9):
            #     rejected_docs["nsfw"].append([record.record_id, nsfw_label, nsfw_conf])
            #     continue

            # toxic_label, toxic_conf = classify_toxic_speech(text)
            # if toxic_label == "toxic" or (toxic_label == "non-toxic" and toxic_conf < 0.8):
            #     rejected_docs["toxic"].append([record.record_id, toxic_label, toxic_conf])
            #     continue

            is_gopher_quality = gopher_quality_filter(text)
            if not is_gopher_quality:
                rejected_docs["gopher_quality"].append([record.record_id])
                continue

            accepted_file.write(text + "\n\n---END_OF_DOC---\n\n")
            accepted_docs.append(record.record_id)

    meta = {
        "total_docs": total_docs,
        "accepted_ct": len(accepted_docs),
        "accepted_docs": accepted_docs,
        "rejected_ct": sum(len(v) for v in rejected_docs.values()),
        "rejected_by_type": {
            "language": len(rejected_docs["language"]),
            "nsfw": len(rejected_docs["nsfw"]),
            "toxic": len(rejected_docs["toxic"]),
            "gopher_quality": len(rejected_docs["gopher_quality"]),
        },
        "rejected_docs": rejected_docs,
    }

    meta_outpath = f"{output_path}.meta.json"
    with open(meta_outpath, "w") as f:
        json.dump(meta, f, indent=4)

    short_meta = {k: v for k, v in meta.items() if k not in ["accepted_docs", "rejected_docs"]}

    return output_path, meta_outpath, short_meta


def main(
    max_files: int = None,
    single: bool = False,
    outdir: str = OUTDIR,
    data_dir: str = DATA_DIR,
    wait: bool = True,
):
    # random.seed(42)

    # Set up the executor
    executor = submitit.AutoExecutor(folder="/data/c-sniderb/a4-leaderboard/slurm_logs")
    max_simultaneous_jobs = 64
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

    outdir_path = outdir

    if not single:
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
                future = executor.submit(process_wet_file, wfp, os.path.join(outdir_path, wet_filename))
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
    else:
        for wfp in tqdm(wet_filepaths, total=len(wet_filepaths)):
            output_file, meta_outpath, short_meta = process_wet_file(
                wfp, os.path.join(outdir_path, str(pathlib.Path(wfp).name)), progress=True
            )
            print(f"Output file written: {output_file}")
            print(f"Meta file written: {meta_outpath}")
            print(f"Short Meta: {short_meta}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-files", type=int, default=None, help="Maximum number of files to process")
    parser.add_argument("--single", action="store_true", help="Whether to use a single thread")
    parser.add_argument("--out-dir", type=str, default=OUTDIR, help="Output directory")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR, help="Data directory")
    parser.add_argument("--no-wait", action="store_true", help="Whether to wait for the jobs to finish")
    args = parser.parse_args()

    wait = not args.no_wait

    main(
        max_files=args.max_files,
        single=args.single,
        outdir=args.out_dir,
        data_dir=args.data_dir,
        wait=wait,
    )
