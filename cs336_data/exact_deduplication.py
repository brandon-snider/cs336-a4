import os
import mmh3


def exact_line_dedupe(input_files: list[os.PathLike], output_directory: os.PathLike):
    unique_lines = {}
    for file in input_files:
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

    for file in input_files:
        file_name = os.path.basename(file)
        with open(file) as f:
            with open(os.path.join(output_directory, file_name), "w") as f_out:
                for line in f:
                    line_hash = mmh3.hash(line)
                    if unique_lines[line_hash]["count"] > 1 or unique_lines[line_hash]["first_file"] != file_name:
                        continue

                    unique_lines[line_hash]["written"] = True
                    f_out.write(line)
