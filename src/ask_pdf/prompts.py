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

