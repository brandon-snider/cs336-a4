"""Minimal script to run the Gopher quality filters on text extracted from a WARC file."""

import argparse
import os
import random
import json

from fastwarc import ArchiveIterator, WarcRecordType
from tqdm import tqdm
from cs336_data.extract_text import extract_text_from_html_bytes
from cs336_data.gopher_quality_filters import gopher_quality_filter

DEFAULT_WARC_PATH = (
    "/data/CC/example.warc.gz" if os.path.exists("/data/CC/example.warc.gz") else "data/CC/example.warc.gz"
)
EST_DOCS_PER_FILE = 27000


def main(warc_path: str, n: int, m: int | None, outpath: str, progress: bool):
    random.seed(42)
    total_docs = 0
    valid_docs = 0
    reservoir = []

    for record in tqdm(
        ArchiveIterator(
            open(warc_path, "rb"),
            record_types=WarcRecordType.response,
            func_filter=lambda r: r.headers.get("WARC-Identified-Payload-Type") == "text/html",
        ),
        total=EST_DOCS_PER_FILE,
        desc="Processing records",
        disable=not progress,
    ):
        text = extract_text_from_html_bytes(record.reader.read())
        if not text.strip():
            continue

        valid = gopher_quality_filter(text)
        total_docs += 1

        if valid:
            valid_docs += 1

        record = {
            "id": record.headers.get("WARC-Record-ID"),
            "url": record.headers.get("WARC-Target-URI"),
            "text": text,
            "valid": valid,
        }

        if len(reservoir) < n:
            reservoir.append(record)
        elif m is None or total_docs < m:
            idx = random.randrange(total_docs)
            if idx < n:
                reservoir[idx] = record
        else:
            break

    if args.progress:
        print(f"Total HTML docs classified: {total_docs} | Valid: {valid_docs}")

    valid_prop = valid_docs / total_docs if total_docs else 0

    output = {
        "metadata": {
            "total_documents": total_docs,
            "valid_documents": valid_docs,
            "valid_proportion": round(valid_prop, 4),
        },
        "samples": [],
    }

    for record in reservoir:
        snippet = record["text"].replace("\n", " ").strip()[:1000]
        output["samples"].append(
            {
                "url": record["url"],
                "valid": record["valid"],
                "snippet": snippet,
            }
        )

    with open(outpath, "w", encoding="utf-8") as out:
        json.dump(output, out, ensure_ascii=False, indent=4)

    print(f"Wrote metadata and {len(reservoir)} sampled records to {outpath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--warc-path", default=DEFAULT_WARC_PATH, help="Path to the GZIP-compressed WARC file")
    parser.add_argument("--n", type=int, default=500, help="Number of random documents to sample for manual comparison")
    parser.add_argument("--m", type=int, default=None, help="Max. docs to consider for random sampling")
    parser.add_argument("--outpath", default="gopher_out.json", help="Path for sampled records and predictions")
    parser.add_argument("--progress", default=True, action="store_true")
    args = parser.parse_args()
    main(args.warc_path, args.n, args.m, args.outpath, args.progress)
