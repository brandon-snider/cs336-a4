import os
import mmh3
from tqdm import tqdm


def exact_line_dedupe(input_files: list[os.PathLike], output_directory: os.PathLike, progress: bool = False):
    unique_lines = {}
    for file in tqdm(input_files, disable=not progress, desc="Indexing files"):
        file_name = os.path.basename(file)

        with open(file) as f:
            for line in f:
                line_hash = mmh3.hash(line)
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
                    line_hash = mmh3.hash(line)
                    if unique_lines[line_hash]["count"] > 1 or unique_lines[line_hash]["first_file"] != file_name:
                        continue

                    unique_lines[line_hash]["written"] = True
                    f_out.write(line)
