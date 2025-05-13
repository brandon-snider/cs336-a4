import argparse
import concurrent.futures
import os
from tqdm import tqdm
import pathlib

from fastwarc.warc import ArchiveIterator, WarcRecordType, is_http
from fastwarc.stream_io import GZipStream, FileStream
from tldextract import TLDExtract

from cs336_data.extract_text import extract_text_from_html_bytes
from cs336_data.gopher_quality_filters import gopher_quality_filter
from cs336_data.harmful_content import classify_nsfw, classify_toxic_speech
from cs336_data.language_identification import identify_language

OUTDIR = "/data/c-sniderb/a4-leaderboard"
DATA_DIR = "/data/CC"


def get_wet_filepaths(data_dir: str) -> list[str]:
    return [
        os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.startswith("CC") and f.endswith(".warc.wet.gz")
    ]


def process_wet_file(input_path: str, output_path: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    stream = GZipStream(FileStream(input_path, "rb"))
    total_docs = 0
    accepted_docs = 0
    accepted_tokens = 0

    rejected_docs = {
        "language": 0,
        "nsfw": 0,
        "toxic": 0,
        "gopher_quality": 0,
    }

    for record in ArchiveIterator(stream):
        total_docs += 1
        if total_docs % 1000 == 0:
            print(
                f"Processed {total_docs} records | {accepted_tokens:,} tokens | {accepted_tokens / total_docs:,.0f} tok/record",
                end="\r",
                flush=True,
            )

        text = extract_text_from_html_bytes(record.reader.read())

        lang, score = identify_language(text)

        if lang != "en":
            rejected_docs["language"] += 1
            continue

        nsfw_label, nsfw_conf = classify_nsfw(text)
        if nsfw_label == "nsfw" or (nsfw_label == "non-nsfw" and nsfw_conf < 0.9):
            rejected_docs["nsfw"] += 1
            continue

        toxic_label, toxic_conf = classify_toxic_speech(text)
        if toxic_label == "toxic" or (toxic_label == "non-toxic" and toxic_conf < 0.9):
            rejected_docs["toxic"] += 1
            continue

        gopher_quality = gopher_quality_filter(text)
        if not gopher_quality:
            rejected_docs["gopher_quality"] += 1
            continue

        # TODO: add other filters here

        accepted_docs += 1
        tokens = tokenizer.encode(text)
        accepted_tokens += len(tokens)

    print(
        f"Processed {total_docs} records, {accepted_docs} accepted, {accepted_tokens:,} tokens",
        end="\r",
        flush=True,
    )

    print(f"Rejected: {rejected_docs}")

    return output_path


def main(max_files: int = None, single: bool = False):
    # Set up the executor
    num_cpus = len(os.sched_getaffinity(0))
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=num_cpus)
    wet_filepaths = get_wet_filepaths(DATA_DIR)
    outdir_path = OUTDIR

    if max_files is not None:
        wet_filepaths = wet_filepaths[:max_files]

    if not single:
        futures = []

        for wfp in wet_filepaths:
            wet_filename = str(pathlib.Path(wfp).name)
            future = executor.submit(process_wet_file, wfp, os.path.join(outdir_path, wet_filename))
            futures.append(future)

        # Iterate over completed futures as they finish, using a progress bar to keep track
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(wet_filepaths),
        ):
            output_file = future.result()
            print(f"Output file written: {output_file}")
    else:
        for wfp in tqdm(wet_filepaths, total=len(wet_filepaths)):
            output_file = process_wet_file(wfp, os.path.join(outdir_path, str(pathlib.Path(wfp).name)))
            print(f"Output file written: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-files", type=int, default=None, help="Maximum number of files to process")
    parser.add_argument("--single", action="store_true", help="Whether to use a single thread")
    args = parser.parse_args()

    main(max_files=args.max_files, single=args.single)
