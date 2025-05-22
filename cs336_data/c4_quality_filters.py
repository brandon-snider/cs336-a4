"""
C4 Quality Filters

Drop pages with:
- Document-level blacklisted content: "lorem ipsum", "{" (rare in non-code pages)

Note: we don't drop documents with fewer than 3 sentences, unlike the paper
- We assume that these will be caught by the Gopher filters (>50 tokens)

Drop lines with:
- Non-punctuation terminators
- Fewer than 5 words
- Line-level blacklisted content (mostly boilerplate)
"""

page_level_blacklist = {"lorem ipsum", "{"}
line_level_blacklist = {
    "javascript",
    "privacy policy",
    "terms of use",
    "cookie policy",
    "uses cookies",
    "use of cookies",
    "use cookies",
    "all rights reserved",
    "terms and conditions",
    "copyright ©",
    "© copyright",
}

short_line_blacklist = {"powered by", "designed by", "theme by", "template by", "website by"}

valid_line_terminators = (".", "!", "?", '"', "'")


def c4_quality_filter(doc: str, verbose: bool = False) -> tuple[bool, str, dict]:
    """
    Filter a document for C4 quality.

    Returns a tuple of (is_good, filtered_doc, metadata).
    """
    doc_lower = doc.lower()

    # Drop documents with document-level blacklisted content
    if any(word in doc_lower for word in page_level_blacklist):
        return False, "", {"reason": "blacklisted"}

    line_meta = {"short": 0, "blacklisted": 0, "invalid_terminator": 0, "kept": 0}

    kept_lines = []
    lines = doc.splitlines()
    for line in lines:
        # Drop lines with fewer than 5 words
        s = line.strip()
        line_word_ct = len(s.split())
        if not s or line_word_ct < 5:
            line_meta["short"] += 1
            continue

        # Drop lines that don't end with a valid terminator
        if not s.endswith(valid_line_terminators):
            line_meta["invalid_terminator"] += 1
            continue

        # Drop lines with line-level blacklisted content
        line_lower = s.lower()
        if any(word in line_lower for word in line_level_blacklist):
            line_meta["blacklisted"] += 1
            continue

        if line_word_ct < 15 and any(word in line_lower for word in short_line_blacklist):
            line_meta["blacklisted"] += 1
            continue

        line_meta["kept"] += 1
        kept_lines.append(line)

    if not kept_lines:
        return False, "", {"reason": "no lines kept", "line_meta": line_meta}

    return True, "\n".join(kept_lines), {"line_meta": line_meta}
