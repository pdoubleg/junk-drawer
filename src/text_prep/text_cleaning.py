import re

# Pre-compile regex patterns
PAGE_NUM_PATTERN = re.compile(r"Page \d+ of \d+")
HYPHEN_WORD_PATTERN = re.compile(r"(\w+)-(\w+)")
PERIOD_COLON_PATTERN = re.compile(r"([.:])([^ \n])")
CLOSE_PAREN_PATTERN = re.compile(r"(\))(?=[^\s])")
OPEN_PAREN_PATTERN = re.compile(r"(?<=[^\s])(\()")
WHITESPACE_PATTERN = re.compile(r"[ \t]+")


def remove_page_numbers(text):
    return re.sub(r"Page \d+ of \d+", "", text)


def remove_multiple_newlines(page_md):
    page_md = re.sub(r"\n\s*\n", "\n\n", page_md)
    return page_md


def fix_page(text):
    """
    Fixes common text problems in the given text.

    Args:
        text (str): The text to fix.

    Returns:
        str: The fixed text.
    """
    watermarks = [
        "SAMPLE",
        "HO 00 03 10 00 Copyright, Insurance Services Office, Inc., 1999",
        "HOMEOWNERS 3 – SPECIAL FORM",
        "HOMEOWNERS",
        "HO 00 03 10 00",
        "Copyright, Insurance Services Office, Inc., 1999",
    ]

    for watermark in watermarks:
        text = text.replace(watermark, "")

    text = remove_page_numbers(text)

    text = re.sub(HYPHEN_WORD_PATTERN, r"\1\2", text)
    text = re.sub(PERIOD_COLON_PATTERN, r"\1 \2", text)
    text = re.sub(CLOSE_PAREN_PATTERN, r"\1 ", text)
    text = re.sub(OPEN_PAREN_PATTERN, r" \1", text)
    text = re.sub(WHITESPACE_PATTERN, " ", text).strip()

    return text


def clean_pages(pages):
    # Clean one page at a time
    for i in range(len(pages)):
        pages[i] = fix_page(pages[i])

    return pages
