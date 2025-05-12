import argparse
import gzip
import os
import random

from warcio.archiveiterator import ArchiveIterator
from cs336_data.common import abs_or_relative_path
from cs336_data.extract_text import extract_text_from_html_bytes

DEFAULT_WARC_PATH = abs_or_relative_path("data/CC/example.warc.gz")
DEFAULT_TRAIN_OUTPUT = "data/wiki/train_negative.txt"


def main():
    parser = argparse.ArgumentParser(description="Randomly sample negative training examples from a WARC file")
    parser.add_argument("--warc-path", default=DEFAULT_WARC_PATH, help="Path to the GZIP-compressed WARC file")
    parser.add_argument("-n", type=int, default=500, help="Number of random samples to take")
    parser.add_argument("-m", type=int, default=None, help="Max. documents to consider for random sampling")
    parser.add_argument("--train-output", default=DEFAULT_TRAIN_OUTPUT, help="Outfile for the training examples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling")
    args = parser.parse_args()

    random.seed(args.seed)
    total_bytes = os.path.getsize(args.warc_path)  # Progress reporting
    total_docs = 0
    reservoir = []

    with gzip.open(args.warc_path, "rb") as stream:
        compressed_stream = stream.fileobj
        for record in ArchiveIterator(stream):
            # Progress update
            if total_docs and total_docs % 1000 == 0:
                read_bytes = compressed_stream.tell()
                pct = read_bytes / total_bytes * 100
                print(f"Processed ~{total_docs} docs ({pct:.2f}% of file)", end="\r", flush=True)

            # Only HTML responses
            if record.rec_type != "response" or not record.http_headers.get_header("Content-Type", "").startswith(
                "text/html"
            ):
                continue

            html_bytes = record.content_stream().read()
            text = extract_text_from_html_bytes(html_bytes)
            if not text.strip():
                continue

            total_docs += 1

            # Reservoir sampling for training set
            if len(reservoir) < args.n:
                reservoir.append(text)
            elif args.m is None or total_docs < args.m:
                idx = random.randrange(total_docs)
                if idx < args.n:
                    reservoir[idx] = text
            else:
                break

    print(f"\nTotal HTML docs considered: {total_docs}")

    # Add sampled entries
    with open(args.train_output, "w", encoding="utf-8") as f:
        for text in reservoir:
            joined_text = text.replace("\n", " ")
            training_example = f"__label__negative {joined_text}\n"
            f.writelines([training_example])

    print(f"Wrote {len(reservoir)} sampled records to {args.train_output}")


if __name__ == "__main__":
    main()
