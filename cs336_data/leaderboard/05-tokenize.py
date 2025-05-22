import argparse
import multiprocessing
import os
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

INPUT_DIR = "/data/c-sniderb/a4-leaderboard/04-classified-thresholded"
OUTPUT_PATH = "/data/c-sniderb/a4-leaderboard/tokenized-thresholded.bin"
TOKENIZER = AutoTokenizer.from_pretrained("gpt2")


def tokenize_and_add_eos(doc: str):
    return TOKENIZER.encode(doc) + [TOKENIZER.eos_token_id]


def main(input_dir: str, output_path: str):
    docs = []
    input_paths = [
        os.path.join(input_dir, filepath)
        for filepath in sorted(os.listdir(input_dir))
        if filepath.endswith(".warc.wet.gz")
    ]

    for filepath in tqdm(input_paths, desc="Reading docs"):
        with open(filepath) as f:
            docs.extend(f.read().split("\n\n---END_OF_DOC---\n\n"))

    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    chunksize = 100
    results = []

    for result in tqdm(
        pool.imap(tokenize_and_add_eos, docs, chunksize=chunksize),
        total=len(docs),
        desc="Tokenizing docs",
    ):
        results.append(result)

    pool.close()
    pool.join()

    all_ids = [token_id for sublist in results for token_id in sublist]
    print(f"Tokenized and encoded {len(input_paths)} docs into {len(all_ids)} tokens")
    ids_array = np.array(all_ids, dtype=np.uint16)
    ids_array.tofile(output_path)
    print(f"Saved tokenized docs to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default=INPUT_DIR)
    parser.add_argument("--output-path", type=str, default=OUTPUT_PATH)
    args = parser.parse_args()

    main(args.input_dir, args.output_path)
