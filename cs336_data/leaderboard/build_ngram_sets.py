from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import os
import re
import unicodedata
import pickle
from tqdm import tqdm

WS_RE = re.compile(r"\s+")
PUNCT_RE = re.compile(r"[^\w\s]")

SIGNATURES_INPATH = "/data/c-sniderb/a4-leaderboard/near-deduped/signatures.pkl"
OUTPUT_DIRECTORY = "/data/c-sniderb/a4-leaderboard/near-deduped/ngram-sets"


def normalize_text(text: str) -> str:
    """Normalize text by:
    - Lowercasing
    - Removing punctuation
    - Normalizing whitespaces
    - Removing accents
    - Applying NFD unicode normalization
    """
    text = text.lower()
    text = PUNCT_RE.sub(" ", text)
    text = WS_RE.sub(" ", text).strip()
    text = unicodedata.normalize("NFD", text)
    return text


def get_ngram_set(text: str, ngrams: int):
    words = text.split()
    return set(" ".join(words[i : i + ngrams]) for i in range(len(words) - ngrams + 1))


def get_file_normalized_ngram_set(file: os.PathLike, ngrams: int):
    with open(file, encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return get_ngram_set(normalize_text(text), ngrams)


def store_ngram_set(inpath: os.PathLike, outpath: os.PathLike, ngrams: int):
    """Store the normalized n‑gram set for a single file."""
    ngram_set = get_file_normalized_ngram_set(inpath, ngrams)

    with open(outpath, "wb") as f:
        pickle.dump(ngram_set, f)

    return outpath


def store_ngram_sets(
    signatures_inpath: os.PathLike,
    output_directory: os.PathLike,
    num_bands: int,
    ngrams: int,
    progress: bool = False,
):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Index of minhash band -> list of filepaths with a matching band
    bands: dict[tuple[int, ...], list[os.PathLike]] = {}

    # Set of pairs of filenames of files that are candidate duplicates based on minhash bands
    candidate_dups: set[tuple[os.PathLike, os.PathLike]] = set()

    with open(signatures_inpath, "rb") as f:
        signatures = pickle.load(f)

    if progress:
        print(f"Loaded {len(signatures)} signatures from {signatures_inpath}")

    for file, minhash in signatures:
        for band in range(num_bands):
            band_minhash = tuple(minhash[band::num_bands])
            if band_minhash not in bands:
                bands[band_minhash] = []
            bands[band_minhash].append(file)

            for other_file in bands[band_minhash]:
                if other_file != file:
                    candidate_dups.add((file, other_file))

    if progress:
        print(f"Found {len(candidate_dups)} candidate duplicates")

    unique_files = set(file for pair in candidate_dups for file in pair)

    print(f"Storing {len(unique_files)} n‑gram sets")

    jobs = os.cpu_count()
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        submit = partial(pool.submit, store_ngram_set, ngrams=ngrams)

        futures = []
        submitted = 0
        skipped = 0

        for inpath in unique_files:
            outpath = os.path.join(output_directory, f"{os.path.basename(inpath)}.pkl")

            if os.path.exists(outpath):
                skipped += 1
                continue

            submitted += 1

            future = submit(inpath, outpath)
            futures.append(future)

        if progress:
            print(f"Total: {len(unique_files)} files, Submitted: {submitted}, Skipped (already exists): {skipped}")

        for future in tqdm(as_completed(futures), total=len(futures), disable=not progress, desc="Storing n‑gram sets"):
            outpath = future.result()
            # print(f"Stored n‑gram set for {outpath}")


def main():
    store_ngram_sets(
        signatures_inpath=SIGNATURES_INPATH,
        output_directory=OUTPUT_DIRECTORY,
        num_bands=10,
        ngrams=5,
        progress=True,
    )


if __name__ == "__main__":
    main()
