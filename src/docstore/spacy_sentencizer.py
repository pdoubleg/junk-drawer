import spacy
from spacy.pipeline import Sentencizer
from spacy.tokens import Doc
from PyPDF2 import PdfFileReader


class DocumentSentencizer:
    def __init__(self, model_path=None):
        if model_path is not None:
            self.nlp = spacy.load(model_path)
        else:
            self.nlp = spacy.blank('en')
            self.nlp.add_pipe('sentencizer')

    @staticmethod
    def extract_text_from_pdf(file_path):
        pdf = PdfFileReader(file_path)
        text = ''
        for page_num in range(pdf.getNumPages()):
            text += pdf.getPage(page_num).extractText()
        return text

    def annotate_sentences(self, text):
        # Implement your annotation logic here
        pass

    def create_docs(self, sentences):
        docs = []
        for sentence, start_end in sentences:
            doc = self.nlp.make_doc(sentence)
            for token in doc:
                if token.i == 0:
                    token.is_sent_start = True
                else:
                    token.is_sent_start = None
            docs.append(doc)
        return docs

    def train(self, docs, n_iter=25):
        sentencizer = self.nlp.get_pipe("sentencizer")
        for i in range(n_iter):
            for doc in docs:
                sentencizer.update([doc], sgd=self.nlp.create_optimizer())

    def save_model(self, path):
        self.nlp.to_disk(path)

    def train_and_save(self, docs, path, n_iter=25):
        self.train(docs, n_iter)
        self.save_model(path)

    def split_into_sentences(self, text):
        doc = self.nlp(text)
        return list(doc.sents)
    
    
# Few shot LLM Prompt:

"You are a language model tasked with analyzing the structure of a few pages from a document and generating Python code to annotate the document " 
"for sentence segmentation suitable for spaCy's `Sentencizer`. The document follows a specific structure, and sentences should not be split at " 
"periods within enumerations or outlined sections. Here are some examples of the document structure: "

'''
[Insert examples of document structure here]

Based on this structure, please generate Python code to annotate the document for sentence segmentation. 
The code should define a function that takes in a string of text and returns a list of tuples. 
Each tuple should contain the sentence as the first element and a dictionary as the second element, 
with 'start' and 'end' keys indicating the start and end indices of the sentence in the text. 

Here are some examples:

[Insert examples of annotated sentences here]

Please generate Python code following the format of the examples above.
'''



#########################
"""
HOM EOW NERS
HO  00 03 10 00
HO  00 03 10 00
Copyr ight, Ins ur anc e Ser vic es  O f f ic e, Inc ., 1999 
Pag e 1 o f  22
HOMEOWNERS 3 Œ SPECIAL FORM
AG REEM ENT
W e willpr ovide the ins u r anc e des c r ibed in this  polic y
inr etur nf or the pr em ium  and c om plianc e with all
applic able pr ovis ions  of  this  polic y.
DEFINIT IONS
A.
Inthis polic y, "you" and "your " r e f e r  to the "nam ed
ins ur ed" s hown in the Dec lar ations  and the s pous e
if  a r e s ident of  the s a m e  hous ehold. "W e", "us "
and "our " r e f e r  to the Com pany pr oviding this  in-
s ur anc e.
B.
Inaddition,c er t ainwor ds  and phr as es  ar e def ined
as  f o llows :
1.
"Air c r af t Liability", "H over c r af t Liability", "Motor
Vehic le Liability" and "W ater c r af t Liability",
s ubj ec t to the pr ovis ions  in \nb.\n below, m ean the
f o llowing:
a.
Liability f o r  "bodily inj u r y" or  "pr oper ty dam -
age" ar is ing out of  the:
(1)
O wner s hip of  s uc h vehic le or  c r
"""

################
"""
# Now, let's manually split this into sentences, taking care to respect the outline structure and enumerations.

sentences = [
    ("HOM EOW NERS", {"start": 0, "end": 12}),
    ("HO  00 03 10 00", {"start": 13, "end": 26}),
    ("HO  00 03 10 00", {"start": 27, "end": 40}),
    ("Copyright, Insurance Services  Office, Inc ., 1999", {"start": 41, "end": 97}),
    ("Pag e 1 o f  22", {"start": 98, "end": 112}),
    ("HOMEOWNERS 3 Œ SPECIAL FORM", {"start": 113, "end": 140}),
    ("AGREEMENT", {"start": 141, "end": 152}),
    ("W e willpr ovide the insurance des c r ibed in this  policy inr etur nf or the pr em ium  and c om plianc e with all applic able pr ovis ions  of  this  polic y.", {"start": 153, "end": 314}),
    ("DEFINIT IONS", {"start": 315, "end": 327}),
    ("A. Inthis polic y, \"you\" and \"your \" r e f e r  to the \"nam ed ins ur ed\" s hown in the Dec lar ations  and the s pous e if  a r e s ident of  the s a m e  hous ehold. \"W e\", \"us \" and \"our \" r e f e r  to the Com pany pr oviding this  in- s ur anc e.", {"start": 328, "end": 514}),
    ("B. Inaddition,c er t ainwor ds  and phr as es  ar e def ined as  f o llows :", {"start": 515, "end": 590}),
    ("1. \"Air c r af t Liability\", \"H over c r af t Liability\", \"Motor Vehic le Liability\" and \"W ater c r af t Liability\", s ubj ec t to the pr ovis ions  in \nb.\n below, m ean the f o llowing:", {"start": 591, "end": 735}),
    ("a. Liability f o r  \"bodily inj u r y\" or  \"pr oper ty dam - age\" ar is ing out of  the:", {"start": 736, "end": 812}),
    ("(1) O wner s hip of  s uc h vehic le or  c r", {"start": 813, "end": 846})
]
"""

"""
RESULT
[('HOM EOW NERS', {'start': 0, 'end': 12}),
 ('HO  00 03 10 00', {'start': 13, 'end': 26}),
 ('HO  00 03 10 00', {'start': 27, 'end': 40}),
 ('Copyr ight, Ins ur anc e Ser vic es  O f f ic e, Inc ., 1999',
  {'start': 41, 'end': 97}),
 ('Pag e 1 o f  22', {'start': 98, 'end': 112}),
 ('HOMEOWNERS 3 Œ SPECIAL FORM', {'start': 113, 'end': 140}),
 ('AG REEM ENT', {'start': 141, 'end': 152}),
 ('W e willpr ovide the ins u r anc e des c r ibed in this  polic y inr etur nf or the pr em ium  and c om plianc e with all applic able pr ovis ions  of  this  polic y.',
  {'start': 153, 'end': 314}),
 ('DEFINIT IONS', {'start': 315, 'end': 327}),
 ('A. Inthis polic y, "you" and "your " r e f e r  to the "nam ed ins ur ed" s hown in the Dec lar ations  and the s pous e if  a r e s ident of  the s a m e  hous ehold. "W e", "us " and "our " r e f e r  to the Com pany pr oviding this  in- s ur anc e.',
  {'start': 328, 'end': 514}),
 ('B. Inaddition,c er t ainwor ds  and phr as es  ar e def ined as  f o llows :',
  {'start': 515, 'end': 590}),
 ('1. "Air c r af t Liability", "H over c r af t Liability", "Motor Vehic le Liability" and "W ater c r af t Liability", s ubj ec t to the pr ovis ions  in \nb.\n below, m ean the f o llowing:',
  {'start': 591, 'end': 735}),
 ('a. Liability f o r  "bodily inj u r y" or  "pr oper ty dam - age" ar is ing out of  the:',
  {'start': 736, 'end': 812}),
 ('(1) O wner s hip of  s uc h vehic le or  c r', {'start': 813, 'end': 846})]

"""

"""
# Now that we have our annotated sentences, we need to convert them into a format that can be used to train the sentencizer. 
# The sentencizer is trained using a Doc object, which is a sequence of Token objects. 
# The is_sent_start attribute of each Token object is set to True if the token starts a sentence, and None otherwise.

Let's convert our sentences into a list of Doc objects. To do this, I will use the nlp object from spaCy, 
which is a processing pipeline. We'll only use the tokenizer from this pipeline to create the Doc objects.

Note: In the actual training process, you would do this for all the documents in your training data. Here,
I'm only doing it for the first few sentences as an example.



import spacy

# Initialize spaCy model
nlp = spacy.blank('en')

def create_docs(nlp, sentences):
    docs = []
    for sentence, start_end in sentences:
        doc = nlp(sentence)
        for token in doc:
            # set is_sent_start to True for the first token in the sentence, and None for all other tokens
            if token.i == 0:
                token.is_sent_start = True
            else:
                token.is_sent_start = None
        docs.append(doc)
    return docs

docs = create_docs(nlp, sentences)
docs[0]  # Displaying the first Doc object to check its structure

"""

"""
# Now that we have the text extracted from your new document, 
# let's apply our trained sentencizer model on this text. We'll load 
# the trained model from the disk, create a Doc object from the text, 
# and then iterate over the sentences in the Doc object.

# Load the trained model from disk
nlp = spacy.load('/mnt/data/sentencizer_model')

# Create a Doc object from the text
doc = nlp(new_pdf_text)

# Extract sentences
sentences = list(doc.sents)

# Display the first 10 sentences
sentences[:10]





"""