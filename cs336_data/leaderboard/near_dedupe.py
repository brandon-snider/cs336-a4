import os
from cs336_data.minhash_deduplication_parallel import minhash_dedupe
import time

DATA_DIR = "/data/c-sniderb/a4-leaderboard/deduped"
NGRAM_CACHE_DIR = "/data/c-sniderb/a4-leaderboard/near-deduped/ngram-sets"
OUTDIR = "/data/c-sniderb/a4-leaderboard/near-deduped"


def main(data_dir: str = DATA_DIR, outdir: str = OUTDIR, ngram_cache_dir: str = NGRAM_CACHE_DIR):
    t0 = time.time()

    print(os.cpu_count())

    # Create outdir if it doesn't exist
    os.makedirs(outdir, exist_ok=True)

    files = set(os.listdir(data_dir))

    # Get all wet files that have a meta file
    wet_filepaths = [os.path.join(data_dir, f) for f in files if f.endswith(".warc.wet.gz")]

    # wet_filepaths = wet_filepaths[:190]

    print(f"Near-deduping {len(wet_filepaths)} files")

    minhash_dedupe(
        input_files=wet_filepaths,
        output_directory=outdir,
        num_hashes=100,
        num_bands=10,
        ngrams=5,
        jaccard_threshold=0.8,
        progress=True,
        signatures_inpath=os.path.join(outdir, "signatures.pkl"),
        signatures_outpath=os.path.join(outdir, "signatures.pkl"),
        ngram_cache_dir=ngram_cache_dir,
    )

    print(f"Near-deduped {len(wet_filepaths)} files in {time.time() - t0} seconds")


if __name__ == "__main__":
    main()
