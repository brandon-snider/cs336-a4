from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing
import os
import random
import re
import shutil
import mmh3
import numpy as np
import unicodedata
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
    progress: bool = False,
):
    signatures = {}
    with ProcessPoolExecutor(max_workers=len(os.sched_getaffinity(0))) as pool:
        submit = partial(pool.submit, build_signature, ngrams=ngrams, num_hashes=num_hashes)
        futures = [submit(p) for p in input_files]
        for future in tqdm(as_completed(futures), total=len(futures), disable=not progress, desc="Building signatures"):
            path, minhash = future.result()
            signatures[path] = minhash
    return signatures


def build_ngram_set(path: os.PathLike, *, ngrams: int):
    """Compute and return the normalized n‑gram set for a single file."""
    return path, get_file_normalized_ngram_set(path, ngrams)


def collect_ngram_sets(
    files: set[os.PathLike],
    *,
    ngrams: int,
    progress: bool = False,
):
    """Parallel map: file → n‑gram set. Yields (path, set) tuples."""
    ngram_sets = {}
    with ProcessPoolExecutor(max_workers=len(os.sched_getaffinity(0))) as pool:
        submit = partial(pool.submit, build_ngram_set, ngrams=ngrams)
        futures = [submit(p) for p in files]
        for future in tqdm(
            as_completed(futures), total=len(futures), disable=not progress, desc="Building n‑gram sets"
        ):
            path, ngram_set = future.result()
            ngram_sets[path] = ngram_set
    return ngram_sets


def minhash_dedupe(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
    progress: bool = False,
):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    bands: dict[tuple[int, ...], list[os.PathLike]] = {}  # Band -> list of files with that band
    candidate_dups: set[tuple[os.PathLike, os.PathLike]] = set()  # Set of pairs of files that are candidate duplicates
    signatures = collect_signatures(input_files, ngrams=ngrams, num_hashes=num_hashes, progress=progress)

    for file, minhash in signatures.items():
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

    # Map from filepath -> set of files that are in the same cluster of duplicates
    clusters: dict[os.PathLike, set[os.PathLike]] = {}

    unique_files = set()
    for f1, f2 in candidate_dups:
        unique_files.add(f1)
        unique_files.add(f2)
    ngram_sets = collect_ngram_sets(unique_files, ngrams=ngrams, progress=progress)

    for f1, f2 in tqdm(candidate_dups, disable=not progress, desc="Testing + clustering"):
        s1, s2 = ngram_sets[f1], ngram_sets[f2]
        if not s1 or not s2:
            continue

        jaccard_similarity = len(s1 & s2) / len(s1 | s2)

        if jaccard_similarity >= jaccard_threshold:
            clusters.setdefault(f1, set()).add(f2)
            clusters[f2] = clusters[f1]

    if progress:
        print(f"Found {len(clusters)} clusters")

    # Collect unique clusters of duplicates
    cluster_set = {frozenset(cluster) for cluster in clusters.values()}
    files_to_write = [f for f in input_files if f not in clusters]
    files_to_write += [random.choice(tuple(cluster)) for cluster in cluster_set]

    for file in tqdm(files_to_write, disable=not progress, desc="Writing output files"):
        dst = os.path.join(output_directory, os.path.basename(file))
        shutil.copy2(file, dst)
