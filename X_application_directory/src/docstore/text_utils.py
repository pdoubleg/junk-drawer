import re
import string
import unicodedata
import pypdf
import spacy


def pdf_to_pages(file):
    pdf = pypdf.PdfReader(file)
    pages = [page.extract_text() for page in pdf.pages]
    return pages


def find_eos_spacy(text):
    nlp = spacy.load("en_core_web_lg")
    doc = nlp(text)
    return [sent.end_char for sent in doc.sents]


def normalize_text(s):
    s = unicodedata.normalize("NFD", s)

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))
