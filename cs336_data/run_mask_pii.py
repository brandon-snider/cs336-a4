"""Minimal script to run the PII masking system on text extracted from a WARC file."""

import argparse
import os
import random
import json

from fastwarc import ArchiveIterator, WarcRecordType
from tqdm import tqdm
from cs336_data.extract_text import extract_text_from_html_bytes
from cs336_data.mask_pii import mask_emails, mask_phone_numbers, mask_ips

DEFAULT_WARC_PATH = (
    "/data/CC/example.warc.gz" if os.path.exists("/data/CC/example.warc.gz") else "data/CC/example.warc.gz"
)
EST_DOCS_PER_FILE = 27000


def main(warc_path: str, n: int, m: int, outpath: str, progress: bool):
    random.seed(42)
    total_docs = 0
    total_replacements = {"email": 0, "phone_number": 0, "ip": 0, "all": 0}
    max_docs = m
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
        original_text = extract_text_from_html_bytes(record.reader.read())
        if not original_text.strip() or len(original_text) > 1000:
            continue

        text = original_text
        text, emails_masked = mask_emails(text)
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
            "id": record.headers.get("WARC-Record-ID"),
            "url": record.headers.get("WARC-Target-URI"),
            "original_text": original_text,
            "text": text,
            "emails_masked": emails_masked,
            "phone_numbers_masked": phone_numbers_masked,
            "ips_masked": ips_masked,
            "all_masked": all_masked,
        }

        # Reservoir sampling for review
        if len(reservoir) < n:
            reservoir.append(record)
        elif max_docs is None or total_docs < max_docs:
            idx = random.randrange(total_docs)
            if idx < n:
                reservoir[idx] = record
        else:
            break

    if progress:
        print(f"Total HTML docs masked: {total_docs}")

    output = {
        "metadata": {
            "total_documents": total_docs,
            "total_replacements": total_replacements,
        },
        "samples": [],
    }

    for record in reservoir:
        output["samples"].append(record)

    with open(outpath, "w", encoding="utf-8") as out:
        json.dump(output, out, ensure_ascii=False, indent=2)

    print(f"Wrote metadata and {len(reservoir)} sampled records to {outpath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--warc-path", default=DEFAULT_WARC_PATH, help="Path to the GZIP-compressed WARC file")
    parser.add_argument("--n", type=int, default=200, help="Number of documents containing PII to process")
    parser.add_argument(
        "--m", type=int, default=None, help="Maximum number of documents to consider for random sampling"
    )
    parser.add_argument("--outpath", default="pii_masking_out.json", help="Path for sampled records")
    parser.add_argument("--progress", default=True, action="store_true")
    args = parser.parse_args()
    main(args.warc_path, args.n, args.m, args.outpath, args.progress)
