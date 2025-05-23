"""Minimal script to run the harmful content classification system on text extracted from a WARC file."""

import argparse
import os
import random
import json

from fastwarc import ArchiveIterator, WarcRecordType
from tqdm import tqdm
from cs336_data.extract_text import extract_text_from_html_bytes
from cs336_data.harmful_content import classify_nsfw, classify_toxic_speech

DEFAULT_WARC_PATH = (
    "/data/CC/example.warc.gz" if os.path.exists("/data/CC/example.warc.gz") else "data/CC/example.warc.gz"
)
EST_DOCS_PER_FILE = 27000


def main(warc_path: str, n: int, outpath: str, progress: bool):
    random.seed(42)
    total_docs = 0
    nsfw_docs = 0
    toxic_docs = 0
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

        if toxic_label == "toxic":
            toxic_docs += 1

        record = {
            "id": record.headers.get("WARC-Record-ID"),
            "url": record.headers.get("WARC-Target-URI"),
            "text": text,
            "nsfw": nsfw_label,
            "nsfw_confidence": nsfw_confidence,
            "toxic": toxic_label,
            "toxic_confidence": toxic_confidence,
        }

        if len(reservoir) < n:
            reservoir.append(record)
        else:
            idx = random.randrange(total_docs)
            if idx < n:
                reservoir[idx] = record

    if progress:
        print(f"Total HTML docs classified: {total_docs} | NSFW: {nsfw_docs} | Toxic: {toxic_docs}")

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

    with open(outpath, "w", encoding="utf-8") as out:
        json.dump(output, out, ensure_ascii=False, indent=2)

    print(f"Wrote metadata and {len(reservoir)} sampled records to {outpath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run harmful content classification on text extracted from a WARC file"
    )
    parser.add_argument("--warc-path", default=DEFAULT_WARC_PATH, help="Path to the GZIP-compressed WARC file")
    parser.add_argument("-n", type=int, default=500, help="Number of random documents to sample for manual comparison")
    parser.add_argument(
        "--outpath", default="harmful_content_out.json", help="Path for sampled records and predictions"
    )
    parser.add_argument("--progress", default=True, action="store_true", help="Report progress during processing")
    args = parser.parse_args()
    main(args.warc_path, args.n, args.outpath, args.progress)
