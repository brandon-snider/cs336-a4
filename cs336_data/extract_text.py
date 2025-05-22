from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding


def extract_text_from_html_bytes(html_bytes: bytes) -> str:
    encoding = detect_encoding(html_bytes)
    unicode_text = html_bytes.decode(encoding, errors="ignore")
    text = extract_plain_text(unicode_text)
    return text


def bytes_to_unicode(seq: bytes) -> str:
    encoding = detect_encoding(seq)
    unicode_text = seq.decode(encoding, errors="ignore")
    return unicode_text
