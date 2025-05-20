import json
import os
import pickle
from collections import defaultdict
from fastwarc import FileStream, GZipStream
from fastwarc.warc import ArchiveIterator, WarcRecordType
import tldextract
from tqdm import tqdm
from transformers import AutoTokenizer
from cs336_data.extract_text import extract_text_from_html_bytes
from cs336_data.gopher_quality_filters import gopher_quality_filter
from cs336_data.harmful_content import classify_nsfw, classify_toxic_speech
from cs336_data.language_identification import identify_language
from cs336_data.leaderboard.common import c4_100_domains


def main():
    data_dir = "/data/c-sniderb/a4-leaderboard/lang-toxic-gopher"
    filepaths = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.startswith("CC") and f.endswith(".warc.wet.gz.meta.json")
    ]

    stats = {
        "total_docs": 0,
        "total_accepted_docs": 0,
        "total_rejected_docs": 0,
        "total_rejected_language": 0,
        "total_rejected_nsfw": 0,
        "total_rejected_toxic": 0,
        "total_rejected_gopher": 0,
    }

    for filepath in filepaths:
        with open(filepath) as f:
            meta = json.load(f)

            stats["total_docs"] += meta["total_docs"]
            stats["total_accepted_docs"] += meta["accepted_ct"]
            stats["total_rejected_docs"] += meta["rejected_ct"]
            stats["total_rejected_language"] += meta["rejected_by_type"]["language"]
            stats["total_rejected_nsfw"] += meta["rejected_by_type"]["nsfw"]
            stats["total_rejected_toxic"] += meta["rejected_by_type"]["toxic"]
            stats["total_rejected_gopher"] += meta["rejected_by_type"]["gopher_quality"]

    print(stats)


def main2():
    data_dir = "/data/c-sniderb/a4-leaderboard/deduped"
    filepaths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".warc.wet.gz")]

    filepath = filepaths[0]

    with open(filepath) as f:
        text = f.read()
        print(text[:1000])


def merge_pickles():
    data_dir = "/data/c-sniderb/a4-leaderboard/near-deduped"
    filepaths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pkl")]

    signatures = []

    for filepath in filepaths:
        with open(filepath, "rb") as f:
            signatures.extend(pickle.load(f))

    outpath = os.path.join(data_dir, "signatures.pkl")

    with open(outpath, "wb") as f:
        pickle.dump(signatures, f)

    print(f"Dumped {len(signatures)} signatures to {outpath}")


if __name__ == "__main__":
    # main()
    # main2()
    # merge_pickles()

    # data_dir = "/data/c-sniderb/a4-leaderboard/near-deduped"
    # ngram_cache_dir = os.path.join(data_dir, "ngram-sets")

    # candidate_dups_path = os.path.join(data_dir, "candidate_dups.pkl")

    # with open(candidate_dups_path, "rb") as f:
    #     candidate_dups = pickle.load(f)

    # i = 0
    # for f1, f2 in candidate_dups:
    #     print(f1, f2)

    #     s1_path = os.path.join(ngram_cache_dir, os.path.basename(f1) + ".pkl")
    #     s2_path = os.path.join(ngram_cache_dir, os.path.basename(f2) + ".pkl")

    #     with open(s1_path, "rb") as f:
    #         s1 = pickle.load(f)

    #     with open(s2_path, "rb") as f:
    #         s2 = pickle.load(f)

    #     jaccard_similarity = len(s1 & s2) / len(s1 | s2)

    #     print(jaccard_similarity)

    #     i += 1
    #     if i > 10:
    #         break

    # DATA_DIR = "/data/CC"

    # filepaths = [
    #     os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.startswith("CC") and f.endswith(".warc.wet.gz")
    # ]

    # records = 0
    # files = 0
    # domains = defaultdict(int)
    # total_tokens = 0

    # tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # for filepath in tqdm(filepaths, desc="Processing files"):
    #     stream = GZipStream(FileStream(filepath, "rb"))
    #     examples = []
    #     for record in ArchiveIterator(stream):
    #         # target_uri = record.headers.get("WARC-Target-URI", None)
    #         # if target_uri is None:
    #         #     continue

    #         # ext = tldextract.extract(target_uri)
    #         # fqdn = ext.fqdn
    #         # tdups = ext.top_domain_under_public_suffix
    #         # key = fqdn if fqdn in c4_100_domains else tdups if tdups in c4_100_domains else None
    #         text = extract_text_from_html_bytes(record.reader.read())
    #         # if key is not None:
    #         #     domains[key] += 1
    #         #     text = extract_text_from_html_bytes(record.reader.read())
    #         #     tokens = tokenizer.encode(text)
    #         #     total_tokens += len(tokens)
    #         # records += 1

    #         # if records > 10:
    #         #     break

    #         lang, score = identify_language(text)
    #         if lang != "en" or score < 0.7:
    #             continue

    #         nsfw_label, nsfw_conf = classify_nsfw(text)
    #         if nsfw_label == "nsfw" or (nsfw_label == "non-nsfw" and nsfw_conf < 0.9):
    #             continue

    #         toxic_label, toxic_conf = classify_toxic_speech(text)
    #         if toxic_label == "toxic" or (toxic_label == "non-toxic" and toxic_conf < 0.8):
    #             continue

    #         is_gopher_quality = gopher_quality_filter(text)
    #         if not is_gopher_quality:
    #             continue

    #         examples.append(text)

    #     print(f"Tokenizing {len(examples)} examples")
    #     tokens = tokenizer.encode(" ".join(examples))
    #     total_tokens += len(tokens)
    #     files += 1

    #     tokens_per_file = total_tokens / files
    #     est_total_tokens = tokens_per_file * len(filepaths)

    #     print(
    #         f"Matched {sum(domains.values()):,} documents from {files} files | {total_tokens:,} tokens | {est_total_tokens:,} tokens"
    #     )
    #     # print(domains)

    #     # if files > 10:
    #     #     break

    # print(records)

    # for filepath in filepaths:
    #     print(filepath)
