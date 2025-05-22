import argparse
import os
import random

from fastwarc import ArchiveIterator
from cs336_data.gopher_quality_filters import gopher_quality_filter
from cs336_data.language_identification import identify_language
from cs336_data.extract_text import bytes_to_unicode
from tqdm import tqdm


DATA_DIR = "/data/c-sniderb/a4-leaderboard/classified-bucketed"
CC_DATA_DIR = "/data/CC"
MAX_FILES = 100

blacklist = {
    "javascript": 0,
    "terms of use": 0,
    "privacy policy": 0,
    "cookie policy": 0,
    "uses cookies": 0,
    "use of cookies": 0,
    "use cookies": 0,
    "{": 0,
    "lorem ipsum": 0,
}

valid_terminators = (".", "!", "?", '"', "'")


def main(data_dir: str = DATA_DIR, max_files: int = MAX_FILES):
    paths = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".warc.wet.gz")]
    random.seed(42)
    paths = random.sample(paths, max_files)

    total_docs = 0
    total_lines = 0
    multiline_docs = 0
    blacklist_hit = 0
    invalid_terminators = 0

    for path in tqdm(paths, desc="Processing files"):
        with open(path, encoding="utf-8", errors="replace") as f:
            docs = f.read().split("\n\n---END_OF_DOC---\n\n")
            for doc in docs:
                total_docs += 1

                lines = doc.split("\n")
                if len(lines) > 1:
                    multiline_docs += 1

                for line in lines:
                    total_lines += 1
                    blacklisted = False
                    line_lower = line.lower()

                    for word in blacklist:
                        if word in line_lower:
                            blacklist[word] += 1
                            blacklisted = True

                    if not line_lower.endswith(valid_terminators):
                        print(line_lower[-10:])
                        invalid_terminators += 1
                        blacklisted = True

                    if blacklisted:
                        blacklist_hit += 1

    print(f"Total docs in {len(paths)} files: {total_docs:,}")
    print(f"Total lines: {total_lines:,}")
    print(f"Multiline docs: {multiline_docs:,} ({multiline_docs / total_docs * 100:.2f}%)")
    print(f"Blacklist hit: {blacklist_hit:,} ({blacklist_hit / total_lines * 100:.2f}%)")
    print(f"Invalid terminators: {invalid_terminators:,} ({invalid_terminators / total_lines * 100:.2f}%)")
    print(blacklist)


def cc():
    paths = [os.path.join(CC_DATA_DIR, f) for f in os.listdir(CC_DATA_DIR) if f.endswith(".warc.wet.gz")]
    random.seed(42)
    paths = random.sample(paths, MAX_FILES)

    files_processed = 0
    total_docs = 0
    total_lines = 0
    multiline_docs = 0
    blacklist_hit = 0
    invalid_terminators = 0

    for path in tqdm(paths, desc="Processing files"):
        files_processed += 1
        for record in tqdm(ArchiveIterator(open(path, "rb")), desc="Processing records", total=27000):
            raw_bytes = record.reader.read()
            doc = bytes_to_unicode(raw_bytes)

            lang, score = identify_language(doc)

            if lang != "en" or score < 0.85:
                continue

            is_gopher_quality = gopher_quality_filter(doc)
            if not is_gopher_quality:
                continue

            total_docs += 1

            lines = doc.split("\n")
            if len(lines) > 1:
                multiline_docs += 1

            for line in lines:
                total_lines += 1
                blacklisted = False
                line_lower = line.lower()

                for word in blacklist:
                    if word in line_lower:
                        blacklist[word] += 1
                        blacklisted = True

                if not line_lower.endswith(valid_terminators):
                    invalid_terminators += 1
                    blacklisted = True

                if blacklisted:
                    blacklist_hit += 1

            if total_docs > 100:
                break

        if total_docs > 100:
            break

        print(f"Total docs in {files_processed} files: {total_docs:,}")
        print(f"Total lines: {total_lines:,}")
        print(f"Multiline docs: {multiline_docs:,} ({multiline_docs / total_docs * 100:.2f}%)")
        print(f"Blacklist hit: {blacklist_hit:,} ({blacklist_hit / total_lines * 100:.2f}%)")
        print(f"Invalid terminators: {invalid_terminators:,} ({invalid_terminators / total_lines * 100:.2f}%)")
        print(blacklist)
        print("-" * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=DATA_DIR)
    parser.add_argument("--max-files", type=int, default=MAX_FILES)
    args = parser.parse_args()
    # main()
    cc()
