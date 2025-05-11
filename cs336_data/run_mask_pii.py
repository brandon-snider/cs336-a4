"""
Minimal script to run the PII masking system on text extracted from a WARC file.
  1. Extracts text from each HTML response record.
  2. Masks PII from the text.
  3. Writes out URL, masked text, and a text snippet to a JSONL file for manual review.
"""

import argparse
import gzip
import os
import random
import json

from warcio.archiveiterator import ArchiveIterator
from cs336_data.extract_text import extract_text_from_html_bytes
from cs336_data.mask_pii import mask_emails, mask_phone_numbers, mask_ips

DEFAULT_WARC_PATH = (
    "/data/CC/example.warc.gz" if os.path.exists("/data/CC/example.warc.gz") else "data/CC/example.warc.gz"
)


def main():
    parser = argparse.ArgumentParser(description="Run PII masking on text extracted from a WARC file")
    parser.add_argument("--warc-path", default=DEFAULT_WARC_PATH, help="Path to the GZIP-compressed WARC file")
    parser.add_argument("-n", type=int, default=40, help="Number of documents containing PII to process")
    parser.add_argument(
        "-m", type=int, default=1000, help="Maximum number of documents to consider for random sampling"
    )
    parser.add_argument(
        "--output",
        default="pii_masking_output.json",
        help="Output JSON file path for sampled records",
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
    total_replacements = {"email": 0, "phone_number": 0, "ip": 0, "all": 0}
    max_docs = args.m

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
            original_text = extract_text_from_html_bytes(html_bytes)
            if not original_text.strip():
                continue

            if len(original_text) > 1000:
                continue

            # Classify every document
            text, emails_masked = mask_emails(original_text)
            text, phone_numbers_masked = mask_phone_numbers(text)
            text, ips_masked = mask_ips(text)
            all_masked = emails_masked + phone_numbers_masked + ips_masked

            total_docs += 1

            if all_masked == 0:
                continue

            total_replacements["email"] += emails_masked
            total_replacements["phone_number"] += phone_numbers_masked
            total_replacements["ip"] += ips_masked
            total_replacements["all"] += all_masked

            record = {
                "url": record.rec_headers.get_header("WARC-Target-URI"),
                "original_text": original_text,
                "text": text,
                "emails_masked": emails_masked,
                "phone_numbers_masked": phone_numbers_masked,
                "ips_masked": ips_masked,
                "all_masked": all_masked,
            }

            # Reservoir sampling for review
            if len(reservoir) < args.n:
                reservoir.append(record)
            elif total_docs < max_docs:
                idx = random.randrange(total_docs)
                if idx < args.n:
                    reservoir[idx] = record
            else:
                break

    if args.progress:
        print(f"\nTotal HTML docs masked: {total_docs}")

    # Prepare output data
    output = {
        "metadata": {
            "total_documents": total_docs,
            "total_replacements": total_replacements,
        },
        "samples": [],
    }

    # Add sampled entries
    for record in reservoir:
        output["samples"].append(record)
    # Write JSON output
    with open(args.output, "w", encoding="utf-8") as out:
        json.dump(output, out, ensure_ascii=False, indent=2)

    print(f"Wrote metadata and {len(reservoir)} sampled records to {args.output}")


if __name__ == "__main__":
    main()
