import argparse
import gzip
from cs336_data.extract_text import extract_text_from_html_bytes

SIZE = 1024 * 10


def main():
    parser = argparse.ArgumentParser(description="Compare text extraction from WARC vs WET files")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--warc", action="store_true", help="Run in WARC mode")
    group.add_argument("--wet", action="store_true", help="Run in WET mode")
    args = parser.parse_args()

    if args.warc:
        path = "/data/CC/example.warc.gz"
        with gzip.open(path, "rb") as f:
            snippet = f.read(SIZE * 10)
        text = extract_text_from_html_bytes(snippet)
    else:
        path = "/data/CC/example.warc.wet.gz"
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
            text = f.read(SIZE)
    print(text)


if __name__ == "__main__":
    main()
