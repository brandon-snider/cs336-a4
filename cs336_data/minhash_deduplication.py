from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import os
import random
import re
import shutil
import mmh3
import numpy as np
import unicodedata
import pickle
from tqdm import tqdm

WS_RE = re.compile(r"\s+")
PUNCT_RE = re.compile(r"[^\w\s]")


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


def get_minhash(ngram_set: set[str], num_perm: int) -> list[int]:
    seeds = np.arange(num_perm, dtype=np.uint32)
    mins = np.full(num_perm, np.iinfo(np.uint32).max, dtype=np.uint32)

    for ngram in ngram_set:
        h = mmh3.hash(ngram, signed=False)
        mins[:] = np.minimum(mins, h ^ seeds)
    return mins.tolist()


def get_ngram_set(text: str, ngrams: int):
    words = text.split()
    return set(" ".join(words[i : i + ngrams]) for i in range(len(words) - ngrams + 1))


def get_file_normalized_ngram_set(file: os.PathLike, ngrams: int):
    with open(file, encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return get_ngram_set(normalize_text(text), ngrams)


def build_signature(path: os.PathLike, *, ngrams: int, num_hashes: int):
    ngram_set = get_file_normalized_ngram_set(path, ngrams)
    return path, get_minhash(ngram_set, num_hashes)


def collect_signatures(
    input_files: list[os.PathLike],
    *,
    ngrams: int,
    num_hashes: int,
    jobs: int | None = None,
    progress: bool = False,
):
    jobs = jobs or os.cpu_count()
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        submit = partial(pool.submit, build_signature, ngrams=ngrams, num_hashes=num_hashes)

        futures = [submit(p) for p in input_files]

        for future in tqdm(as_completed(futures), total=len(futures), disable=not progress, desc="Building signatures"):
            yield future.result()


def build_ngram_set(path: os.PathLike, *, ngrams: int):
    """Compute and return the normalized n‑gram set for a single file."""
    return path, get_file_normalized_ngram_set(path, ngrams)


def collect_ngram_sets(
    files: set[os.PathLike],
    *,
    ngrams: int,
    jobs: int | None = None,
    progress: bool = False,
):
    """Parallel map: file → n‑gram set. Yields (path, set) tuples."""
    jobs = jobs or os.cpu_count()
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        submit = partial(pool.submit, build_ngram_set, ngrams=ngrams)
        futures = [submit(p) for p in files]
        for future in tqdm(
            as_completed(futures), total=len(futures), disable=not progress, desc="Building n‑gram sets"
        ):
            yield future.result()


def minhash_dedupe(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
    progress: bool = False,
    signatures_inpath: os.PathLike | None = None,
    signatures_outpath: os.PathLike | None = None,
):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Index of minhash band -> list of filepaths with a matching band
    bands: dict[tuple[int, ...], list[os.PathLike]] = {}

    # Set of pairs of filenames of files that are candidate duplicates based on minhash bands
    candidate_dups: set[tuple[os.PathLike, os.PathLike]] = set()

    if signatures_inpath is not None and os.path.exists(signatures_inpath):
        with open(signatures_inpath, "rb") as f:
            signatures = pickle.load(f)
        print(f"Loaded {len(signatures)} signatures from {signatures_inpath}")
    else:
        signatures = list(collect_signatures(input_files, ngrams=ngrams, num_hashes=num_hashes, progress=progress))

        if signatures_outpath is not None:
            with open(signatures_outpath, "wb") as f:
                pickle.dump(signatures, f)
            print(f"Dumped {len(signatures)} signatures to {signatures_outpath}")

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
        print(f"Will test + cluster {len(candidate_dups)} candidate duplicates")

    unique_files = {f for pair in candidate_dups for f in pair}
    ngram_sets = dict(collect_ngram_sets(unique_files, ngrams=ngrams, progress=progress))

    # Map from filepath -> set of files that are in the same cluster of duplicates
    clusters: dict[os.PathLike, set[os.PathLike]] = {}

    for f1, f2 in tqdm(candidate_dups, disable=not progress, desc="Test + cluster candidate dupes"):
        s1, s2 = ngram_sets[f1], ngram_sets[f2]

        jaccard_similarity = len(s1 & s2) / len(s1 | s2)

        if jaccard_similarity >= jaccard_threshold:
            if f1 not in clusters:
                clusters[f1] = {f1}
            if f2 not in clusters:
                clusters[f2] = clusters[f1]
            clusters[f1].add(f2)

    # Set of clusters of duplicates
    cluster_set: set[frozenset[os.PathLike]] = {frozenset(cluster) for cluster in clusters.values()}

    # List of output files: unique files, and one at random from each cluster of duplicates
    files_to_write: list[os.PathLike] = [f for f in input_files if f not in clusters]
    for cluster in cluster_set:
        files_to_write.append(random.choice(list(cluster)))

    for file in tqdm(files_to_write, disable=not progress, desc="Writing output files"):
        dst = os.path.join(output_directory, os.path.basename(file))
        shutil.copy2(file, dst)
