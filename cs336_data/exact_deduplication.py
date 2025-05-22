import os
import mmh3
from tqdm import tqdm


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


def exact_line_dedupe_docs(
    input_files: list[os.PathLike],
    output_directory: os.PathLike,
    progress: bool = False,
):
    counts: dict[int, int] = {}
    for path in tqdm(input_files, disable=not progress, desc="Counting"):
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if not line.strip():
                    continue
                h = mmh3.hash(line.rstrip("\n"), signed=False)
                counts[h] = counts.get(h, 0) + 1

    for path in tqdm(input_files, disable=not progress, desc="Rewrite"):
        out_path = os.path.join(output_directory, os.path.basename(path))
        with (
            open(path, encoding="utf-8", errors="replace", newline=None) as fin,
            open(out_path, "w", encoding="utf-8") as fout,
        ):
            content = fin.read()
            docs = content.split("\n\n---END_OF_DOC---\n\n")

            for doc in docs:
                buf: list[str] = []
                content_written = False

                lines = doc.split("\n")

                for i, line in enumerate(lines):
                    if i < len(lines) - 1:
                        line = line + "\n"

                    if not line.strip():
                        buf.append(line)
                        continue

                    h = mmh3.hash(line.rstrip("\n"), signed=False)

                    if h not in counts:
                        print(doc)
                        print("-" * 100)
                        print(line)
                        print(doc == line)

                    if counts[h] == 1:
                        if not content_written:
                            content_written = True
                        buf.append(line)

                if content_written:
                    fout.writelines(buf)
                    fout.write("\n\n---END_OF_DOC---\n\n")
