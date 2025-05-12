import os


def abs_or_relative_path(path: str):
    if os.path.exists("/" + path):
        return "/" + path

    if os.path.exists(path):
        return path

    raise FileNotFoundError(f"Could not find {path}")
