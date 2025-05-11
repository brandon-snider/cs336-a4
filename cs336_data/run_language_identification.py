"""
Minimal script to run the language identification system on text extracted from a WARC file.
Implements part (c) of the `language_identification` problem by:
  1. Extracting text from each HTML response record.
  2. Sampling a reproducible set of n documents.
  3. Predicting language and confidence for each sample.
  4. Writing out URL, predicted language, score, and a text snippet to a JSONL file for manual review.
"""

import argparse
import gzip
import os
import random
import json

from warcio.archiveiterator import ArchiveIterator
from cs336_data.extract_text import extract_text_from_html_bytes
from cs336_data.language_identification import identify_language

DEFAULT_WARC_PATH = "/data/CC/example.warc.gz"


def main():
    parser = argparse.ArgumentParser(description="Run language identification on text extracted from a WARC file")
    parser.add_argument("--warc-path", default=DEFAULT_WARC_PATH, help="Path to the GZIP-compressed WARC file")
    parser.add_argument("-n", type=int, default=200, help="Number of random documents to sample for manual comparison")
    parser.add_argument(
        "--output",
        default="language_identification_output.json",
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
    english_docs = 0

    # Reservoir sampling storage
    reservoir = []

    with gzip.open(args.warc_path, "rb") as stream:
        compressed_stream = stream.fileobj
        for record in ArchiveIterator(stream):
            # Progress update
            if args.progress and total_docs and total_docs % 1000 == 0:
                read_bytes = compressed_stream.tell()
                pct = read_bytes / total_bytes * 100
                print(f"Processed ~{total_docs} docs ({pct:.2f}% of file)", end="\r", flush=True)

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
            lang, score = identify_language(text)
            total_docs += 1
            if lang == "en":
                english_docs += 1

            # Reservoir sampling for review
            if len(reservoir) < args.n:
                reservoir.append((text, record.rec_headers.get_header("WARC-Target-URI"), lang, score))
            else:
                idx = random.randrange(total_docs)
                if idx < args.n:
                    reservoir[idx] = (text, record.rec_headers.get_header("WARC-Target-URI"), lang, score)

    if args.progress:
        print(f"\nTotal HTML docs classified: {total_docs}")

    # Compute English proportion
    eng_prop = english_docs / total_docs if total_docs else 0

    # Prepare output data
    output = {
        "metadata": {
            "total_documents": total_docs,
            "english_documents": english_docs,
            "english_proportion": round(eng_prop, 4),
        },
        "samples": [],
    }

    # Add sampled entries
    for text, url, lang, score in reservoir:
        snippet = text.replace("\n", " ").strip()[:1000]
        output["samples"].append({"url": url, "lang": lang, "score": round(score, 4), "snippet": snippet})

    # Write JSON output
    with open(args.output, "w", encoding="utf-8") as out:
        json.dump(output, out, ensure_ascii=False, indent=2)

    print(f"Wrote metadata and {len(reservoir)} sampled records to {args.output}")


if __name__ == "__main__":
    main()
