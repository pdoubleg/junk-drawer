from io import BytesIO
import re
from typing import List
import spacy
from llama_index.schema import TextNode
from llama_index import Document



def parse_pdf(file: BytesIO) -> List[str]:
    pdf = PdfReader(file)
    output = []
    for page in pdf.pages:
        text = page.extract_text()
        # Merge hyphenated words
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        # Fix newlines in the middle of sentences
        text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
        # Remove multiple newlines
        text = re.sub(r"\n\s*\n", "\n\n", text)

        output.append(text)

    return output


def find_eos_spacy(text):
    nlp = spacy.load('en_core_web_lg')
    doc = nlp(text)
    return [sent.end_char for sent in doc.sents]


def split_pages_into_chunks(pages, chunk_size):
    """
    Split pages (list of texts) into smaller fragments (list of texts).
    """
    page_offset = [0]
    for _, page in enumerate(pages):
        page_offset += [page_offset[-1]+len(page)+1]

    if chunk_size:
        text = ' '.join(pages)
        return text_to_chunks(text, chunk_size, page_offset)
    else:
        return pages


def text_to_chunks(text, size, page_offset):
    """
    Split single text into smaller fragments (list of texts).
    """
    if size and len(text) > size:
        out = []
        pos = 0
        page = 1
        p_off = page_offset.copy()[1:]
        eos = find_eos_spacy(text)
        if len(text) not in eos:
            eos += [len(text)]
        for i in range(len(eos)):
            if eos[i] - pos > size:
                text_chunk = text[pos:eos[i]]
                out += [text_chunk]
                pos = eos[i]
                if eos[i] > p_off[0]:
                    page += 1
                    del p_off[0]
        # ugly: last chunk
        text_chunk = text[pos:eos[i]]
        out += [text_chunk]
        out = [x for x in out if x]
        return out
    else:
        return [text]
    
    
def create_nodes_from_chunks(chunks, form_name, form_id):
    nodes = []
    for i in range(len(chunks)):
        node = TextNode(
            text=chunks[i], 
            id_=f"node_id_{[i]}",
            metadata={
                'form_name': form_name,
                'form_id': form_id,
            },
        )
        if i > 0:
            node.relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(node_id=nodes[i-1].node_id)
            nodes[i-1].relationships[NodeRelationship.NEXT] = RelatedNodeInfo(node_id=node.node_id)
    
        nodes.append(node)
    
    return nodes


def create_relationships(nodes, i):
    if i > 0:
        node = nodes[i]
        previous_node = nodes[i-1]
        
        node.relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(node_id=previous_node.node_id)
        previous_node.relationships[NodeRelationship.NEXT] = RelatedNodeInfo(node_id=node.node_id)

    


def parse_pdf_fitz(path: Path, docname: str, chunk_chars: int, overlap: int) -> List[str]:
    import fitz

    file = fitz.open(path)
    split = ""
    pages: List[str] = []
    nodes: List[Node] = []
    for i in range(file.page_count):
        page = file.load_page(i)
        split += page.get_text("text", sort=True)
        pages.append(str(i + 1))
        # split could be so long it needs to be split
        # into multiple chunks. Or it could be so short
        # that it needs to be combined with the next chunk.
        while len(split) > chunk_chars:
            # pretty formatting of pages (e.g. 1-3, 4, 5-7)
            pg = "-".join([pages[0], pages[-1]])
            nodes.append(
                TextNode(
                    text=split[:chunk_chars], 
                    metadata={'filename': f"{docname}",
                              'pages': f"{pg}"},
                )
            )
            split = split[chunk_chars - overlap :]
            pages = [str(i + 1)]
    if len(split) > overlap:
        pg = "-".join([pages[0], pages[-1]])
        nodes.append(
                TextNode(
                    text=split[:chunk_chars], 
                    metadata={'filename': f"{docname}",
                              'pages': f"{pg}"},
                )
            )
    file.close()
    
    return nodes


def parse_pdf(path: Path, docname: str, chunk_chars: int, overlap: int) -> List[str]:
    import pypdf
    pdfFileObj = open(path, "rb")
    pdfReader = pypdf.PdfReader(pdfFileObj)
    split = ""
    pages: List[str] = []
    nodes: List[str] = []
    for i, page in enumerate(pdfReader.pages):
        split += page.extract_text()
        pages.append(str(i + 1))
        # split could be so long it needs to be split
        # into multiple chunks. Or it could be so short
        # that it needs to be combined with the next chunk.
        while len(split) > chunk_chars:
            # pretty formatting of pages (e.g. 1-3, 4, 5-7)
            pg = "-".join([pages[0], pages[-1]])
            nodes.append(
                    TextNode(
                        text=split[:chunk_chars], 
                        metadata={'filename': f"{docname}",
                                'pages': f"{pg}"},
                    )
                )
            split = split[chunk_chars - overlap :]
            pages = [str(i + 1)]
    if len(split) > overlap:
        pg = "-".join([pages[0], pages[-1]])
        nodes.append(
                TextNode(
                    text=split[:chunk_chars], 
                    metadata={'filename': f"{docname}",
                             'pages': f"{pg}"},
                )
            )
    pdfFileObj.close()
    return nodes




    
