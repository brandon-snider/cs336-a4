import math
from nltk.tokenize import word_tokenize

MIN_TOKENS = 50
MAX_TOKENS = 100000

MIN_MEAN_TOKEN_LENGTH = 3
MAX_MEAN_TOKEN_LENGTH = 10

MAX_ELIPSIS_LINE_RATIO = 0.3

MIN_ALPHA_TOKENS_RATIO = 0.8


def gopher_quality_filter(text: str, verbose: bool = False) -> bool:
    tokens = word_tokenize(text)

    if len(tokens) < MIN_TOKENS or len(tokens) > MAX_TOKENS:
        if verbose:
            print(f"Too few or too many words: word count = {len(tokens)}")
        return False

    mean_tok_len = sum(len(token) for token in tokens) / len(tokens)
    if mean_tok_len < MIN_MEAN_TOKEN_LENGTH or mean_tok_len > MAX_MEAN_TOKEN_LENGTH:
        if verbose:
            print(f"Mean token length too short/long: mean token length = {mean_tok_len}")
        return False

    lines = text.split("\n")
    elipsis_line_ct = sum(1 for line in lines if line.endswith("..."))
    elipsis_line_ratio = elipsis_line_ct / len(lines)
    if elipsis_line_ratio > MAX_ELIPSIS_LINE_RATIO:
        if verbose:
            print(f"Too many lines ending with elipsis: ratio = {elipsis_line_ratio}")
        return False

    min_alpha_tokens = math.ceil(len(tokens) * MIN_ALPHA_TOKENS_RATIO)
    max_non_alpha_tokens = len(tokens) - min_alpha_tokens
    ct_non_alpha_tokens = 0
    for token in tokens:
        if not any(char.isalpha() for char in token):
            ct_non_alpha_tokens += 1

        if ct_non_alpha_tokens > max_non_alpha_tokens:
            if verbose:
                print("Too many tokens with zero alphabetic characters (stopped early)")
            return False

    return True


if __name__ == "__main__":
    # res = gopher_quality_filter("Hello, world!...\nSomething else.")
    text = ("This should definitely be valid input text and of high quality according to Gopher rules. ") * 100
    print(gopher_quality_filter(text))
