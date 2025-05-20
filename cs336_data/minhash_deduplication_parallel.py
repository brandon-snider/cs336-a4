from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
import gc
import os
import random
import re
import shutil
import mmh3
import numpy as np
import unicodedata
import pickle
import submitit
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


def collect_ngram_sets(files: set[os.PathLike], ngrams: int, progress: bool = False):
    executor = submitit.AutoExecutor(folder="/data/c-sniderb/a4-leaderboard/slurm_logs")
    max_simultaneous_jobs = 64

    executor.update_parameters(
        slurm_array_parallelism=max_simultaneous_jobs,
        timeout_min=5,
        mem_gb=2,
        cpus_per_task=1,
        slurm_account="student",
        slurm_partition="a4-cpu",
        slurm_qos="a4-cpu-qos",
    )

    futures = []
    results = {}

    with executor.batch():
        for file in files:
            future = executor.submit(build_ngram_set, file, ngrams=ngrams)
            futures.append(future)

    for future in tqdm(
        submitit.helpers.as_completed(futures), total=len(futures), disable=not progress, desc="Building n‑gram sets"
    ):
        path, ngram_set = future.result()
        results[path] = ngram_set

    return results


# def collect_ngram_sets(
#     files: set[os.PathLike],
#     *,
#     ngrams: int,
#     jobs: int | None = None,
#     progress: bool = False,
# ):
#     """Parallel map: file → n‑gram set. Yields (path, set) tuples."""
#     jobs = jobs or os.cpu_count()
#     with ProcessPoolExecutor(max_workers=jobs) as pool:
#         submit = partial(pool.submit, build_ngram_set, ngrams=ngrams)
#         futures = [submit(p) for p in files]
#         for future in tqdm(
#             as_completed(futures), total=len(futures), disable=not progress, desc="Building n‑gram sets"
#         ):
#             yield future.result()


def minhash_dedupe(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
    progress: bool = False,
    batch_pairs: int = 10000,
    cache_size: int = 3000,
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

    # Map from filepath -> set of files that are in the same cluster of duplicates
    clusters: dict[os.PathLike, set[os.PathLike]] = {}
    ngram_cache: OrderedDict[os.PathLike, set[str]] = OrderedDict()

    pairs_list = sorted(candidate_dups)

    for start in tqdm(
        range(0, len(pairs_list), batch_pairs), desc="Testing + clustering candidate dupes", disable=not progress
    ):
        chunk = pairs_list[start : start + batch_pairs]
        files_needed = {f for pair in chunk for f in pair}
        hits = {p: ngram_cache[p] for p in files_needed if p in ngram_cache}

        for p in hits:
            ngram_cache.move_to_end(p)

        misses = files_needed - hits.keys()

        if progress:
            print(f"Hits: {len(hits)} | Misses: {len(misses)} | Cache: {len(ngram_cache)}")

        surplus = max(0, len(ngram_cache) + len(misses) - cache_size)
        for _ in range(surplus):
            ngram_cache.popitem(last=False)

        new_sets = dict(collect_ngram_sets(misses, ngrams=ngrams, progress=progress)) if misses else {}

        for p, s in new_sets.items():
            ngram_cache[p] = s

        ngram_sets = {**hits, **new_sets}

        def is_dup(pair):
            f1, f2 = pair
            s1, s2 = ngram_sets[f1], ngram_sets[f2]
            if not s1 or not s2:
                return None
            jaccard_similarity = len(s1 & s2) / len(s1 | s2)
            return pair if jaccard_similarity >= jaccard_threshold else None

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as tpool:
            for dup in tqdm(tpool.map(is_dup, chunk), desc="Testing + clustering chunk", disable=not progress):
                if dup:
                    f1, f2 = dup
                    clusters.setdefault(f1, set()).add(f2)
                    clusters[f2] = clusters[f1]

        del ngram_sets, new_sets
        gc.collect()

    # Set of clusters of duplicates
    cluster_set: set[frozenset[os.PathLike]] = {frozenset(cluster) for cluster in clusters.values()}
    files_to_write = [f for f in input_files if f not in clusters] + [random.choice(list(c)) for c in cluster_set]

    for file in tqdm(files_to_write, disable=not progress, desc="Writing output files"):
        dst = os.path.join(output_directory, os.path.basename(file))
        shutil.copy2(file, dst)
