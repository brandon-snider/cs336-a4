import gzip
import random

from cs336_data.common import abs_or_relative_path

MAX_URLS = 1e1
MAX_TO_PROCESS = 1e5

INPATH = abs_or_relative_path("data/wiki/enwiki-20240420-extracted_urls.txt.gz")
OUTPATH = "data/wiki/subsampled_positive_urls.txt"


def sample_positive_urls(inpath: str, outpath: str, max_urls: int = 1e1, max_to_process: int = 1e5):
    reservoir = []
    random.seed(42)

    with gzip.open(inpath, "rt") as f:
        processed = 0

        for line in f:
            processed += 1
            if processed % 1e6 == 0:
                print(f"Processed {processed:,} lines", end="\r")

            url = line.strip()
            if not url:
                continue

            if len(reservoir) < MAX_URLS:
                reservoir.append(url)
            else:
                r = random.randint(0, len(reservoir))
                if r < MAX_URLS:
                    reservoir[r] = url

            if processed >= MAX_TO_PROCESS:
                break

    with open(outpath, "w") as f:
        for url in reservoir:
            f.write(url + "\n")

    print(f"Wrote {len(reservoir)} URLs to {outpath}")


if __name__ == "__main__":
    sample_positive_urls(INPATH, OUTPATH, MAX_URLS, MAX_TO_PROCESS)
