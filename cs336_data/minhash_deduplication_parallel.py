from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
import gc
import os
import random
import re
import shutil
import time
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


def load_ngram_set(
    filepath: os.PathLike, *, ngram_cache_dir: os.PathLike | None = None, ngrams: int, progress: bool = False
):
    filename = os.path.basename(filepath)
    ngram_set = None
    inpath = None if ngram_cache_dir is None else os.path.join(ngram_cache_dir, filename + ".pkl")

    if inpath is not None:
        if os.path.exists(inpath):
            with open(inpath, "rb") as f:
                ngram_set = pickle.load(f)
        elif progress:
            print(f"Cache miss for {filepath}")

    if ngram_set is None:
        ngram_set = get_file_normalized_ngram_set(filepath, ngrams)
        if inpath is not None:
            with open(inpath, "wb") as f:
                pickle.dump(ngram_set, f)

    return ngram_set


def compare_pair(
    f1: os.PathLike,
    f2: os.PathLike,
    *,
    ngram_cache_dir: os.PathLike | None = None,
    ngrams: int,
    jaccard_threshold: float,
    progress: bool = False,
):
    s1 = load_ngram_set(f1, ngram_cache_dir=ngram_cache_dir, ngrams=ngrams, progress=progress)
    s2 = load_ngram_set(f2, ngram_cache_dir=ngram_cache_dir, ngrams=ngrams, progress=progress)

    print(s1, s2)

    if not s1 or not s2:
        print(f"Skipping {f1} and {f2} because one or both have no n‑gram set")
        return False

    jaccard_similarity = len(s1 & s2) / len(s1 | s2)

    if jaccard_similarity >= jaccard_threshold:
        return True
    return False


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
    ngram_cache_dir: os.PathLike | None = None,
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

    with open(os.path.join(output_directory, "candidate_dups.pkl"), "wb") as f:
        pickle.dump(candidate_dups, f)

    if progress:
        print(f"Will test + cluster {len(candidate_dups)} candidate duplicates")

    # Map from filepath -> set of files that are in the same cluster of duplicates
    clusters: dict[os.PathLike, set[os.PathLike]] = {}

    jobs = os.cpu_count()
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        submit = partial(
            pool.submit,
            compare_pair,
            ngram_cache_dir=ngram_cache_dir,
            ngrams=ngrams,
            jaccard_threshold=jaccard_threshold,
        )

        futures = []

        for f1, f2 in candidate_dups:
            future = submit(f1, f2)
            futures.append(future)

        i = 0
        print_every = 100

        for future in tqdm(
            as_completed(futures), total=len(futures), disable=not progress, desc="Testing + clustering"
        ):
            if future.result():
                clusters.setdefault(f1, set()).add(f2)
                clusters[f2] = clusters[f1]

            i += 1
            if i % print_every == 0:
                print(f"Clusters: {len(clusters)}")

    if progress:
        print(f"Found {len(clusters)} clusters")

    # Set of clusters of duplicates
    cluster_set: set[frozenset[os.PathLike]] = {frozenset(cluster) for cluster in clusters.values()}
    files_to_write = [f for f in input_files if f not in clusters] + [random.choice(list(c)) for c in cluster_set]

    for file in tqdm(files_to_write, disable=not progress, desc="Writing output files"):
        dst = os.path.join(output_directory, os.path.basename(file))
        shutil.copy2(file, dst)
