"""
A module to clean ISO HO3 Homeowners policy document used for experimentation.
Cleans PDF pages one at a time so that page number references are maintained.
Source document: https://www.iii.org/sites/default/files/docs/pdf/HO3_sample.pdf
"""

import re

# Pre-compile regex patterns
PAGE_NUM_PATTERN = re.compile(r"Page \d+ of \d+")
HYPHEN_WORD_PATTERN = re.compile(r"(\w+)-(\w+)")
PERIOD_COLON_PATTERN = re.compile(r"([.:])([^ \n])")
CLOSE_PAREN_PATTERN = re.compile(r"(\))(?=[^\s])")
OPEN_PAREN_PATTERN = re.compile(r"(?<=[^\s])(\()")
WHITESPACE_PATTERN = re.compile(r"[ \t]+")


# We will track page numbers using the pdf file
def remove_page_numbers(text):
    """Removes page number footer from the text"""
    return re.sub(r"Page \d+ of \d+", "", text)


def clean_sample_ho3_pages(text):
    """
    Fixes various text issues for this public policy form.

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

