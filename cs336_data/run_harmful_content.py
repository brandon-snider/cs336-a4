"""
Minimal script to run the harmful content classification system on text extracted from a WARC file.
Implements part (d) of the `harmful_content` problem by:
  1. Extracting text from each HTML response record.
  2. Sampling a reproducible set of n documents.
  3. Predicting harmful content and confidence for each sample.
  4. Writing out URL, predicted classification, score, and a text snippet to a JSONL file for manual review.
"""

import argparse
import gzip
import os
import random
import json

from warcio.archiveiterator import ArchiveIterator
from cs336_data.extract_text import extract_text_from_html_bytes
from cs336_data.harmful_content import classify_nsfw, classify_toxic_speech

DEFAULT_WARC_PATH = (
    "/data/CC/example.warc.gz" if os.path.exists("/data/CC/example.warc.gz") else "data/CC/example.warc.gz"
)


def main():
    parser = argparse.ArgumentParser(
        description="Run harmful content classification on text extracted from a WARC file"
    )
    parser.add_argument("--warc-path", default=DEFAULT_WARC_PATH, help="Path to the GZIP-compressed WARC file")
    parser.add_argument("-n", type=int, default=500, help="Number of random documents to sample for manual comparison")
    # parser.add_argument(
    #     "-m", type=int, default=1000, help="Maximum number of documents to consider for random sampling"
    # )
    parser.add_argument(
        "--output",
        default="harmful_content_output.json",
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
    nsfw_docs = 0
    toxic_docs = 0

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
                    f"Processed ~{total_docs} docs ({pct:.2f}% of file) | NSFW: {nsfw_docs} | Toxic: {toxic_docs}",
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
            nsfw_label, nsfw_confidence = classify_nsfw(text)
            toxic_label, toxic_confidence = classify_toxic_speech(text)
            total_docs += 1

            if nsfw_label == "non-nsfw" and nsfw_confidence < 0.9:
                nsfw_label = "nsfw"
                nsfw_confidence = 1 - nsfw_confidence

            if toxic_label == "non-toxic" and toxic_confidence < 0.9:
                toxic_label = "toxic"
                toxic_confidence = 1 - toxic_confidence

            if nsfw_label == "non-nsfw" and toxic_label == "non-toxic":
                continue

            if nsfw_label == "nsfw":
                nsfw_docs += 1
            elif nsfw_label != "non-nsfw":
                print(f"NSFW label: {nsfw_label} | Confidence: {nsfw_confidence}")

            if toxic_label == "toxic":
                toxic_docs += 1
            elif toxic_label != "non-toxic":
                print(f"Toxic label: {toxic_label} | Confidence: {toxic_confidence}")

            record = {
                "url": record.rec_headers.get_header("WARC-Target-URI"),
                "text": text,
                "nsfw": nsfw_label,
                "nsfw_confidence": nsfw_confidence,
                "toxic": toxic_label,
                "toxic_confidence": toxic_confidence,
            }

            # Reservoir sampling for review
            if len(reservoir) < args.n:
                reservoir.append(record)
            else:
                idx = random.randrange(total_docs)
                if idx < args.n:
                    reservoir[idx] = record

    if args.progress:
        print(f"\nTotal HTML docs classified: {total_docs} | NSFW: {nsfw_docs} | Toxic: {toxic_docs}")

    nsfw_prop = nsfw_docs / total_docs if total_docs else 0
    toxic_prop = toxic_docs / total_docs if total_docs else 0

    # Prepare output data
    output = {
        "metadata": {
            "total_documents": total_docs,
            "nsfw_documents": nsfw_docs,
            "nsfw_proportion": round(nsfw_prop, 4),
            "toxic_documents": toxic_docs,
            "toxic_proportion": round(toxic_prop, 4),
        },
        "samples": [],
    }

    # Add sampled entries
    for record in reservoir:
        snippet = record["text"].replace("\n", " ").strip()[:1000]
        output["samples"].append(
            {
                "url": record["url"],
                "nsfw": record["nsfw"],
                "nsfw_confidence": round(record["nsfw_confidence"], 4),
                "toxic": record["toxic"],
                "toxic_confidence": round(record["toxic_confidence"], 4),
                "snippet": snippet,
            }
        )

    # Write JSON output
    with open(args.output, "w", encoding="utf-8") as out:
        json.dump(output, out, ensure_ascii=False, indent=2)

    print(f"Wrote metadata and {len(reservoir)} sampled records to {args.output}")


if __name__ == "__main__":
    main()
