import re
from pypdf import PdfReader


def convert_pdf_to_text(file_path):
    pdf_reader = PdfReader(file_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text
def remove_footers(text):
    text = re.sub(r'Page \d+ of \d+ Copyright, Insurance Services Office, Inc., \d+', '', text)
    return text
def split_text_into_sections(text):
    sections = re.split(r'(?<=[a-z])\n(?=[A-Z])', text)
    return sections
def save_sections_as_txt_files(sections):
    for i, section in enumerate(sections):
        if len(section) > 2000:
            chunks = [section[i:i+2000] for i in range(0, len(section), 2000)]
            for j, chunk in enumerate(chunks):
                with open(f"section_{i}_{j}.txt", "w") as f:
                    f.write(chunk)
        else:
            with open(f"section_{i}.txt", "w") as f:
                f.write(section)
