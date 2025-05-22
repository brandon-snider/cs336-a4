import os
import random

from fastwarc import ArchiveIterator
from cs336_data.extract_text import extract_text_from_html_bytes


DATA_DIR = "/data/c-sniderb/a4-leaderboard/lang-gopher"
CC_DATA_DIR = "/data/CC"


def main():
    metapaths = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".meta.json")]

    random.seed(42)
    random.shuffle(metapaths)

    cc_path = os.path.join(CC_DATA_DIR, "CC-MAIN-20250417135010-20250417165010-00015.warc.wet.gz")

    # target_record = "<urn:uuid:4656ef77-1e13-4875-a63d-20dd82299a4c>" # Example 1: language (Korean)
    # target_record = "<urn:uuid:5d946215-72de-4a60-abf5-a5a5de823a00>"  # Example 2: Gopher
    # ... (dedupe example)
    target_record = "<urn:uuid:24ebfc47-2fdd-454d-bacd-ce11cdd84c5a>"  # Example 4: Gopher
    # ... (classifier example)

    for record in ArchiveIterator(open(cc_path, "rb"), func_filter=lambda x: x.record_id == target_record):
        text = extract_text_from_html_bytes(record.reader.read())
        print(text)
        break


if __name__ == "__main__":
    main()
