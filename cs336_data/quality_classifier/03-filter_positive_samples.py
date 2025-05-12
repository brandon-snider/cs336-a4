import argparse
import gzip
import os

from warcio.archiveiterator import ArchiveIterator
from warcio.warcwriter import WARCWriter

from cs336_data.common import abs_or_relative_path
from cs336_data.extract_text import extract_text_from_html_bytes
from cs336_data.language_identification import identify_language
from cs336_data.harmful_content import classify_nsfw, classify_toxic_speech
from cs336_data.gopher_quality_filters import gopher_quality_filter

DEFAULT_WARC_PATH = abs_or_relative_path("data/wiki/unfiltered_positive_samples.warc.gz")
DEFAULT_OUTPUT_PATH = "data/wiki/positive_samples.warc.gz"
DEFAULT_TRAINING_SAMPLES_PATH = "data/wiki/train_positive.txt"


def main():
    parser = argparse.ArgumentParser(description="Filter samples from a WARC file")
    parser.add_argument("--warc-path", default=DEFAULT_WARC_PATH, help="Path to the GZIP-compressed WARC file")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Path to the output WARC file")
    parser.add_argument(
        "--train-outpath", default=DEFAULT_TRAINING_SAMPLES_PATH, help="Where the training examples should be written"
    )
    args = parser.parse_args()

    # Prepare for progress reporting
    total_bytes = os.path.getsize(args.warc_path)

    # Counters for all records
    total_docs = 0
    passed_docs = 0
    rejected_docs = {
        "language": 0,
        "nsfw": 0,
        "toxic": 0,
        "gopher_quality": 0,
    }

    positive_samples = []
    training_samples = []

    with gzip.open(args.warc_path, "rb") as stream:
        compressed_stream = stream.fileobj
        for record in ArchiveIterator(stream):
            # Progress update
            if total_docs and total_docs % 1000 == 0:
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

            total_docs += 1

            print(text[:100])

            # Classify every document
            lang, score = identify_language(text)

            if lang != "en":
                rejected_docs["language"] += 1
                continue

            nsfw_label, nsfw_conf = classify_nsfw(text)
            if nsfw_label == "nsfw":
                rejected_docs["nsfw"] += 1
                continue

            toxic_label, toxic_conf = classify_toxic_speech(text)
            if toxic_label == "toxic":
                rejected_docs["toxic"] += 1
                continue

            gopher_quality = gopher_quality_filter(text)
            if not gopher_quality:
                rejected_docs["gopher_quality"] += 1
                continue

            passed_docs += 1

            positive_samples.append(record)

            joined_text = text.replace("\n", " ")
            training_sample = f"__label__positive {joined_text}\n"
            training_samples.append(training_sample)

    print(f"\nTotal HTML docs processed: {total_docs} | Passed: {passed_docs} | Rejected: {total_docs - passed_docs}")
    print(rejected_docs)

    with gzip.open(args.output, "wb") as stream:
        writer = WARCWriter(stream, gzip=True)
        for record in positive_samples:
            writer.write_record(record)

    with open(args.train_outpath, "w") as f:
        f.writelines(training_samples)

    print(f"Wrote {len(positive_samples)} positive samples to {args.output}")
    print(f"Wrote {len(training_samples)} training samples to {args.train_outpath}")


if __name__ == "__main__":
    main()
