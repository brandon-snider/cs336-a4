import os
import random

from fastwarc import ArchiveIterator
from cs336_data.extract_text import extract_text_from_html_bytes


DATA_DIR = "/data/c-sniderb/a4-leaderboard/01-english"
CC_DATA_DIR = "/data/CC"


def main():
    metapaths = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".meta.json")]

    random.seed(42)
    random.shuffle(metapaths)

    cc_path = os.path.join(CC_DATA_DIR, "CC-MAIN-20250417135010-20250417165010-00023.warc.wet.gz")

    target_record = "<urn:uuid:7b1a75e4-af02-40dc-82b4-20e2540dc98f>"  # Example 1: language (German)
    # target_record = "<urn:uuid:5d946215-72de-4a60-abf5-a5a5de823a00>"  # Example 2: Gopher
    # ... (dedupe example)
    # target_record = "<urn:uuid:24ebfc47-2fdd-454d-bacd-ce11cdd84c5a>"  # Example 4: Gopher
    # ... (classifier example)

    for record in ArchiveIterator(open(cc_path, "rb"), func_filter=lambda x: x.record_id == target_record):
        text = extract_text_from_html_bytes(record.reader.read())
        print(text)
        break


if __name__ == "__main__":
    main()
