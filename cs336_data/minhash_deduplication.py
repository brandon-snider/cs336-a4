import os
import random
import re
from collections.abc import Callable
import shutil
import unicodedata
import mmh3


def normalize_text(text: str) -> str:
    """Normalize text by:
    - Lowercasing
    - Removing punctuation
    - Normalizing whitespaces
    - Removing accents
    - Applying NFD unicode normalization
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    text = unicodedata.normalize("NFD", text)
    return text


def get_hash_function(seed: int):
    def func(text: str):
        return mmh3.hash(text, seed)

    return func


def get_hash_functions(num_hashes: int):
    return [get_hash_function(i) for i in range(num_hashes)]


def get_minhash(ngram_set: set[str], hash_functions: list[Callable[[str], int]]):
    return [min(hash_function(ngram) for ngram in ngram_set) for hash_function in hash_functions]


def get_ngram_set(text: str, ngrams: int):
    words = text.split()
    return set(" ".join(words[i : i + ngrams]) for i in range(len(words) - ngrams + 1))


def get_file_normalized_ngram_set(file: os.PathLike, ngrams: int):
    with open(file) as f:
        text = f.read()
    return get_ngram_set(normalize_text(text), ngrams)


def minhash_dedupe(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Index of minhash band -> list of filepaths with a matching band
    bands: dict[tuple[int, ...], list[os.PathLike]] = {}

    # Set of pairs of filenames of files that are candidate duplicates based on minhash bands
    candidate_dups: set[tuple[os.PathLike, os.PathLike]] = set()

    hash_functions = get_hash_functions(num_hashes)

    for file in input_files:
        ngram_set = get_file_normalized_ngram_set(file, ngrams)
        minhash = get_minhash(ngram_set, hash_functions)

        for band in range(num_bands):
            band_minhash = tuple(minhash[band::num_bands])
            if band_minhash not in bands:
                bands[band_minhash] = []
            bands[band_minhash].append(file)

            for other_file in bands[band_minhash]:
                if other_file != file:
                    candidate_dups.add((file, other_file))

    # Map from filepath -> set of files that are in the same cluster of duplicates
    clusters: dict[os.PathLike, set[os.PathLike]] = {}

    for candidate_dup in candidate_dups:
        f1, f2 = candidate_dup

        ngram_set_1 = get_file_normalized_ngram_set(f1, ngrams)
        ngram_set_2 = get_file_normalized_ngram_set(f2, ngrams)

        jaccard_similarity = len(ngram_set_1.intersection(ngram_set_2)) / len(ngram_set_1.union(ngram_set_2))

        if jaccard_similarity >= jaccard_threshold:
            if f1 not in clusters:
                clusters[f1] = {f1}
            if f2 not in clusters:
                clusters[f2] = clusters[f1]
            clusters[f1].add(f2)

    # Set of clusters of duplicates
    cluster_set: set[frozenset[os.PathLike]] = set()

    for cluster in clusters.values():
        cluster_set.add(frozenset(cluster))

    # List of output files: unique files, and one at random from each cluster of duplicates
    files_to_write: list[os.PathLike] = []

    for file in input_files:
        if file not in clusters:
            # File is not in a cluster, so it is a non-duplicate
            files_to_write.append(file)

    for cluster in cluster_set:
        # Pick a random file from a cluster of duplicates
        random_file = random.choice(list(cluster))
        files_to_write.append(random_file)

    for file in files_to_write:
        dst = os.path.join(output_directory, os.path.basename(file))
        shutil.copy2(file, dst)
