from langchain.prompts import PromptTemplate

## Use a shorter template to reduce the number of tokens in the prompt
template = """Create a final answer to the given questions using the provided document excerpts(in no particular order) as references. ALWAYS include a "SOURCES" section in your answer including only the minimal set of sources needed to answer the question. If you are unable to answer the question, simply state that you do not know. Do not attempt to fabricate an answer and leave the SOURCES section empty.

---------

QUESTION: What  is the purpose of ARPA-H?
=========
Content: More support for patients and families. \n\nTo get there, I call on Congress to fund ARPA-H, the Advanced Research Projects Agency for Health. \n\nIt’s based on DARPA—the Defense Department project that led to the Internet, GPS, and so much more.  \n\nARPA-H will have a singular purpose—to drive breakthroughs in cancer, Alzheimer’s, diabetes, and more.
Source: 1-32
Content: While we’re at it, let’s make sure every American can get the health care they need. \n\nWe’ve already made historic investments in health care. \n\nWe’ve made it easier for Americans to get the care they need, when they need it. \n\nWe’ve made it easier for Americans to get the treatments they need, when they need them. \n\nWe’ve made it easier for Americans to get the medications they need, when they need them.
Source: 1-33
Content: The V.A. is pioneering new ways of linking toxic exposures to disease, already helping  veterans get the care they deserve. \n\nWe need to extend that same care to all Americans. \n\nThat’s why I’m calling on Congress to pass legislation that would establish a national registry of toxic exposures, and provide health care and financial assistance to those affected.
Source: 1-30
=========
FINAL ANSWER: The purpose of ARPA-H is to drive breakthroughs in cancer, Alzheimer’s, diabetes, and more.
SOURCES: 1-32

---------

QUESTION: {question}
=========
{summaries}
=========
FINAL ANSWER:"""

STUFF_PROMPT = PromptTemplate(
    template=template, input_variables=["summaries", "question"]
)


TASK = {
	'v6': (
			"Answer the question truthfully based on the text below. "
			"Include verbatim quote and a comment where to find it in the text (page number). "
			"After the quote write a step by step explanation. "
			"Use bullet points. "
		),
	
	'v4':
		"Answer the question truthfully based on the text below. " \
		"Include verbatim quote and a comment where to find it in the text (ie name of the section and page number). " \
		"After the quote write an explanation (in the new paragraph) for a young reader.",
	'v3': 'Answer the question truthfully based on the text below. Include verbatim quote and a comment where to find it in the text (ie name of the section and page number).',
	'v2': 'Answer question based on context. The answers sould be elaborate and based only on the context.',
	'v1': 'Answer question based on context.',
	'v5':
		"Generate a comprehensive and informative answer for a given question solely based on the provided document fragments. " \
		"You must only use information from the provided fragments. Use an unbiased and journalistic tone. Combine fragments together into coherent answer. Do not repeat text.  " \
		"Following the answer, use bullet point format. Cite fragments using their respective outline notation.  " \
		"Only cite the most relevant fragments that answer the question accurately.",
}

HYDE = "Write an example answer to the following question in the style of an insurance contract. " \
        "Don't write generic answer, just assume the context is a typical property and casualty insurance policy document."
