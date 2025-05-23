"""Minimal script to run the language identification system on text extracted from a WARC file."""

import argparse
import os
import random
import json

from fastwarc import ArchiveIterator, WarcRecordType
from tqdm import tqdm
from cs336_data.extract_text import extract_text_from_html_bytes
from cs336_data.language_identification import identify_language

DEFAULT_WARC_PATH = (
    "/data/CC/example.warc.gz" if os.path.exists("/data/CC/example.warc.gz") else "data/CC/example.warc.gz"
)
EST_DOCS_PER_FILE = 27000


def main(warc_path: str, n: int, outpath: str, progress: bool):
    random.seed(42)
    total_docs = 0
    english_docs = 0
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

        lang, score = identify_language(text)
        total_docs += 1
        if lang == "en":
            english_docs += 1

        if len(reservoir) < n:
            reservoir.append((text, record.headers.get("WARC-Target-URI"), lang, score))
        else:
            idx = random.randrange(total_docs)
            if idx < n:
                reservoir[idx] = (text, record.headers.get("WARC-Target-URI"), lang, score)

    if progress:
        print(f"Total HTML docs classified: {total_docs}")

    eng_prop = english_docs / total_docs if total_docs else 0

    output = {
        "metadata": {
            "total_documents": total_docs,
            "english_documents": english_docs,
            "english_proportion": round(eng_prop, 4),
        },
        "samples": [],
    }

    for text, url, lang, score in reservoir:
        snippet = text.replace("\n", " ").strip()[:1000]
        output["samples"].append({"url": url, "lang": lang, "score": round(score, 4), "snippet": snippet})

    with open(outpath, "w", encoding="utf-8") as out:
        json.dump(output, out, ensure_ascii=False, indent=2)

    print(f"Wrote metadata and {len(reservoir)} sampled records to {outpath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--warc-path", default=DEFAULT_WARC_PATH, help="Path to the GZIP-compressed WARC file")
    parser.add_argument("--n", type=int, default=200, help="Number of random documents to sample for manual comparison")
    parser.add_argument(
        "--outpath", default="language_identification_out.json", help="Path for sampled records and predictions"
    )
    parser.add_argument("--progress", default=True, action="store_true")
    args = parser.parse_args()
    main(args.warc_path, args.n, args.outpath, args.progress)
