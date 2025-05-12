"""
Minimal script to run the Gopher quality filters on text extracted from a WARC file.
Implements part (b) of the `gopher_quality_filters` problem by:
  1. Extracting text from each HTML response record.
  2. Applying the Gopher quality filters to each document.
  3. Writing out URL, predicted classification, and a text snippet to a JSON file for manual review.
"""

import argparse
import gzip
import os
import random
import json

from warcio.archiveiterator import ArchiveIterator
from cs336_data.extract_text import extract_text_from_html_bytes
from cs336_data.gopher_quality_filters import gopher_quality_filter

DEFAULT_WARC_PATH = (
    "/data/CC/example.warc.gz" if os.path.exists("/data/CC/example.warc.gz") else "data/CC/example.warc.gz"
)


def main():
    parser = argparse.ArgumentParser(description="Run Gopher quality filters on text extracted from a WARC file")
    parser.add_argument("--warc-path", default=DEFAULT_WARC_PATH, help="Path to the GZIP-compressed WARC file")
    parser.add_argument("-n", type=int, default=500, help="Number of random documents to sample for manual comparison")
    parser.add_argument(
        "-m", type=int, default=None, help="Maximum number of documents to consider for random sampling"
    )
    parser.add_argument(
        "--output",
        default="gopher_quality_filters_output.json",
        help="Output JSON file path for sampled records and predictions",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling")
    parser.add_argument("--progress", default=True, action="store_true", help="Report progress during processing")
    args = parser.parse_args()

    # Ensure reproducibility
    random.seed(args.seed)

    # Prepare for progress reporting
    total_bytes = os.path.getsize(args.warc_path) if args.progress else None

    # Counters for all records
    total_docs = 0
    valid_docs = 0

    # Reservoir sampling storage
    reservoir = []

    with gzip.open(args.warc_path, "rb") as stream:
        compressed_stream = stream.fileobj
        for record in ArchiveIterator(stream):
            # Progress update
            if args.progress and total_docs and total_docs % 1000 == 0:
                read_bytes = compressed_stream.tell()
                pct = read_bytes / total_bytes * 100
                print(
                    f"Processed ~{total_docs} docs ({pct:.2f}% of file) | Valid: {valid_docs}",
                    end="\r",
                    flush=True,
                )

            # Only HTML responses
            if record.rec_type != "response":
                continue
            ctype = record.http_headers.get_header("Content-Type", "")
            if not ctype.startswith("text/html"):
                continue

            html_bytes = record.content_stream().read()
            text = extract_text_from_html_bytes(html_bytes)
            if not text.strip():
                continue

            # Classify every document
            valid = gopher_quality_filter(text)
            total_docs += 1

            if valid:
                valid_docs += 1

            record = {
                "url": record.rec_headers.get_header("WARC-Target-URI"),
                "text": text,
                "valid": valid,
            }

            # Reservoir sampling for review
            if len(reservoir) < args.n:
                reservoir.append(record)
            elif args.m is None or total_docs < args.m:
                idx = random.randrange(total_docs)
                if idx < args.n:
                    reservoir[idx] = record
            else:
                break

    if args.progress:
        print(f"\nTotal HTML docs classified: {total_docs} | Valid: {valid_docs}")

    valid_prop = valid_docs / total_docs if total_docs else 0

    # Prepare output data
    output = {
        "metadata": {
            "total_documents": total_docs,
            "valid_documents": valid_docs,
            "valid_proportion": round(valid_prop, 4),
        },
        "samples": [],
    }

    # Add sampled entries
    for record in reservoir:
        snippet = record["text"].replace("\n", " ").strip()[:1000]
        output["samples"].append(
            {
                "url": record["url"],
                "valid": record["valid"],
                "snippet": snippet,
            }
        )

    # Write JSON output
    with open(args.output, "w", encoding="utf-8") as out:
        json.dump(output, out, ensure_ascii=False, indent=4)

    print(f"Wrote metadata and {len(reservoir)} sampled records to {args.output}")


if __name__ == "__main__":
    main()
