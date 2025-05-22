import os
import mmh3
from tqdm import tqdm
import concurrent.futures
import pickle


def exact_line_dedupe(input_files: list[os.PathLike], output_directory: os.PathLike, progress: bool = False):
    unique_lines = {}
    for file in tqdm(input_files, disable=not progress, desc="Indexing files"):
        file_name = os.path.basename(file)

        with open(file) as f:
            for line in f:
                line_hash = mmh3.hash(line, signed=False)
                if line_hash not in unique_lines:
                    unique_lines[line_hash] = {
                        "first_file": file_name,
                        "count": 0,
                        "written": False,
                    }
                unique_lines[line_hash]["count"] += 1

    for file in tqdm(input_files, disable=not progress, desc="Writing output files"):
        file_name = os.path.basename(file)
        with open(file) as f:
            with open(os.path.join(output_directory, file_name), "w") as f_out:
                for line in f:
                    line_hash = mmh3.hash(line, signed=False)
                    if unique_lines[line_hash]["count"] > 1 or unique_lines[line_hash]["first_file"] != file_name:
                        continue

                    unique_lines[line_hash]["written"] = True
                    f_out.write(line)


def get_counts_for_file(path: os.PathLike) -> dict[int, int]:
    counts: dict[int, int] = {}

    with open(path) as f:
        docs = f.read().split("\n\n---END_OF_DOC---\n\n")

        for doc in docs:
            lines = doc.splitlines(keepends=True)
            for line in lines:
                s = line.strip()
                if s:
                    h = mmh3.hash(s, signed=False)
                    counts[h] = counts.get(h, 0) + 1

    return counts


def rewrite_file(path: os.PathLike, counts: dict[int, int], output_directory: os.PathLike) -> tuple[int, int]:
    total_lines = 0
    unique_lines = 0

    out_path = os.path.join(output_directory, os.path.basename(path))
    with (
        open(path) as fin,
        open(out_path, "w") as fout,
    ):
        docs = fin.read().split("\n\n---END_OF_DOC---\n\n")

        for doc in docs:
            buf: list[str] = []
            content_written = False
            lines = doc.splitlines(keepends=True)

            for line in lines:
                total_lines += 1

                s = line.strip()
                if not s:
                    buf.append(line)
                    continue

                h = mmh3.hash(s, signed=False)

                if h not in counts or counts[h] == 1:
                    content_written = True
                    buf.append(line)

            est_total_words = sum(len(line.split()) for line in buf)

            if content_written and est_total_words > 50:
                unique_lines += len(buf)
                fout.writelines(buf)
                fout.write("\n\n---END_OF_DOC---\n\n")

    return total_lines, unique_lines


def rewrite_files(paths: list[os.PathLike], counts_path: os.PathLike, output_directory: os.PathLike):
    with open(counts_path, "rb") as f:
        counts = pickle.load(f)

    total_lines = 0
    unique_lines = 0

    for path in paths:
        total_lines_file, unique_lines_file = rewrite_file(path, counts, output_directory)
        total_lines += total_lines_file
        unique_lines += unique_lines_file

    return total_lines, unique_lines


def exact_line_dedupe_docs(
    input_files: list[os.PathLike],
    output_directory: os.PathLike,
    progress: bool = False,
    mp: bool = False,
) -> tuple[int, int]:
    counts: dict[int, int] = {}

    total_lines = 0
    unique_lines = 0

    if mp:
        num_cpus = len(os.sched_getaffinity(0))
        print(f"Using {num_cpus} CPUs")
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=num_cpus)

        futures = []

        for path in input_files:
            future = executor.submit(get_counts_for_file, path)
            futures.append(future)

        for future in tqdm(futures, total=len(input_files), desc="Counting", disable=not progress):
            file_counts = future.result()
            for h, count in file_counts.items():
                counts[h] = counts.get(h, 0) + count

        # Remove counts of unique lines (to save memory/time writing and loading)
        dup_counts = {}
        for k, v in counts.items():
            if v > 1:
                dup_counts[k] = v
        counts = dup_counts

        counts_path = os.path.join(output_directory, "counts.pkl")
        with open(counts_path, "wb") as f:
            pickle.dump(counts, f)

        futures = []
        chunk_size = max(1, len(input_files) // num_cpus)

        for i in range(0, len(input_files), chunk_size):
            chunk = input_files[i : i + chunk_size]
            futures.append(executor.submit(rewrite_files, chunk, counts_path, output_directory))

        for future in tqdm(futures, total=len(input_files), desc="Writing output files", disable=not progress):
            total_lines_file, unique_lines_file = future.result()
            total_lines += total_lines_file
            unique_lines += unique_lines_file

        return total_lines, unique_lines
    else:
        for path in tqdm(input_files, disable=not progress, desc="Counting"):
            file_counts = get_counts_for_file(path)
            for h, count in file_counts.items():
                counts[h] = counts.get(h, 0) + count

        for path in tqdm(input_files, disable=not progress, desc="Writing output files"):
            total_lines_file, unique_lines_file = rewrite_file(path, counts, output_directory)
            total_lines += total_lines_file
            unique_lines += unique_lines_file

        return total_lines, unique_lines
