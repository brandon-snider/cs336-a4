import re

EMAIL_REGEX = re.compile(r"[a-zA-Z0-9.!#$%&\'*+/=?^_`{|}~-]+@(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}")

PHONE_REGEX = re.compile(
    r"""
    (?<!\w)                          # Don't match if preceded by a word character
    (?:\+1[\s.-]?)?                  # Optional country code
    \(?\d{3}\)?                      # Area code with or without parentheses
    [\s.-]?                          # Optional separator
    \d{3}                            # First 3 digits
    [\s.-]?                          # Optional separator
    \d{4}                            # Last 4 digits
    (?!\w)                           # Don't match if followed by a word character
""",
    re.VERBOSE,
)

IPV4_REGEX = re.compile(
    r"""
    (?<!\d)                              # Don't match if preceded by a digit
    (?:
        (?:25[0-5]|                     # 250-255
        2[0-4]\d|                       # 200-249
        1\d{2}|                         # 100-199
        [1-9]?\d)                       # 0-99
        \.
    ){3}
    (?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d) # Final octet
    (?!\d)                              # Don't match if followed by a digit
""",
    re.VERBOSE,
)


def mask_emails(text: str, include_original: bool = False) -> tuple[str, int]:
    placeholder = "|||EMAIL_ADDRESS|||"
    matches = list(EMAIL_REGEX.finditer(text))
    redacted_text = EMAIL_REGEX.sub(placeholder, text)
    return redacted_text, len(matches)


def mask_phone_numbers(text: str) -> tuple[str, int]:
    placeholder = "|||PHONE_NUMBER|||"
    matches = list(PHONE_REGEX.finditer(text))
    redacted_text = PHONE_REGEX.sub(placeholder, text)
    return redacted_text, len(matches)


def mask_ips(text: str) -> tuple[str, int]:
    placeholder = "|||IP_ADDRESS|||"
    matches = list(IPV4_REGEX.finditer(text))
    redacted_text = IPV4_REGEX.sub(placeholder, text)
    return redacted_text, len(matches)
