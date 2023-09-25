from dotenv import load_dotenv
load_dotenv()
from math import exp
import openai
import pandas as pd
from tenacity import retry, wait_random_exponential, stop_after_attempt
import tiktoken

def rerank_with_openai(df: pd.DataFrame,
                       query: str, 
                       text_col_name: str = "text",
                       model_name: str = 'text-davinci-003',
    ) -> pd.DataFrame:
    # Get encodings for our target classes
    tokens = [" Yes", " No"]
    tokenizer = tiktoken.encoding_for_model(model_name)
    ids = [tokenizer.encode(token) for token in tokens]
    ids[0], ids[1]

    prompt = '''
    You are an Assistant responsible for helping detect whether the retrieved document is relevant to the query. For a given input, you need to output a single token: "Yes" or "No" indicating the retrieved document is relevant to the query.

    Query: How to plant a tree?
    Document: """Cars were invented in 1886, when German inventor Carl Benz patented his Benz Patent-Motorwagen.[3][4][5] Cars became widely available during the 20th century. One of the first cars affordable by the masses was the 1908 Model T, an American car manufactured by the Ford Motor Company. Cars were rapidly adopted in the US, where they replaced horse-drawn carriages.[6] In Europe and other parts of the world, demand for automobiles did not increase until after World War II.[7] The car is considered an essential part of the developed economy."""
    Relevant: No

    Query: Has the coronavirus vaccine been approved?
    Document: """The Pfizer-BioNTech COVID-19 vaccine was approved for emergency use in the United States on December 11, 2020."""
    Relevant: Yes

    Query: What is the capital of France?
    Document: """Paris, France's capital, is a major European city and a global center for art, fashion, gastronomy and culture. Its 19th-century cityscape is crisscrossed by wide boulevards and the River Seine. Beyond such landmarks as the Eiffel Tower and the 12th-century, Gothic Notre-Dame cathedral, the city is known for its cafe culture and designer boutiques along the Rue du Faubourg Saint-Honoré."""
    Relevant: Yes

    Query: What are some papers to learn about PPO reinforcement learning?
    Document: """Proximal Policy Optimization and its Dynamic Version for Sequence Generation: In sequence generation task, many works use policy gradient for model optimization to tackle the intractable backpropagation issue when maximizing the non-differentiable evaluation metrics or fooling the discriminator in adversarial learning. In this paper, we replace policy gradient with proximal policy optimization (PPO), which is a proved more efficient reinforcement learning algorithm, and propose a dynamic approach for PPO (PPO-dynamic). We demonstrate the efficacy of PPO and PPO-dynamic on conditional sequence generation tasks including synthetic experiment and chit-chat chatbot. The results show that PPO and PPO-dynamic can beat policy gradient by stability and performance."""
    Relevant: Yes

    Query: Explain sentence embeddings
    Document: """Inside the bubble: exploring the environments of reionisation-era Lyman emitting galaxies with JADES and FRESCO: We present a study of the environments of 16 Lyman emitting galaxies (LAEs) in the reionisation era (5.8<z<8) identified by JWST/NIRSpec as part of the JWST Advanced Deep Extragalactic Survey (JADES). Unless situated in sufficiently (re)ionised regions, Lyman emission from these galaxies would be strongly absorbed by neutral gas in the intergalactic medium (IGM). We conservatively estimate sizes of the ionised regions required to reconcile the relatively low Lyman velocity offsets (ΔvLy<300kms1) with moderately high Lyman escape fractions (fesc,Ly>5%) observed in our sample of LAEs, indicating the presence of ionised ``bubbles'' with physical sizes of the order of 0.1pMpc≲Rion≲1pMpc in a patchy reionisation scenario where the bubbles are embedded in a fully neutral IGM. Around half of the LAEs in our sample are found to coincide with large-scale galaxy overdensities seen in FRESCO at z5.8-5.9 and z7.3, suggesting Lyman transmission is strongly enhanced in such overdense regions, and underlining the importance of LAEs as tracers of the first large-scale ionised bubbles. Considering only spectroscopically confirmed galaxies, we find our sample of UV-faint LAEs (MUV≳−20mag) and their direct neighbours are generally not able to produce the required ionised regions based on the Lyman-α transmission properties, suggesting lower-luminosity sources likely play an important role in carving out these bubbles. These observations demonstrate the combined power of JWST multi-object and slitless spectroscopy in acquiring a unique view of the early stages of Cosmic Reionisation via the most distant LAEs."""
    Relevant: No

    Query: {query}
    Document: """{document}"""
    Relevant:
    '''

    @retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
    def document_relevance(query, document):
        response = openai.Completion.create(
            model=model_name,
            prompt=prompt.format(query=query, document=document),
            temperature=0,
            logprobs=1,
            logit_bias={3363: 1, 1400: 1},
        )

        return (
            query,
            document,
            response["choices"][0]["text"],
            response["choices"][0]["logprobs"]["token_logprobs"][0],
        )
    
    output_list = []
    for index, row in df.iterrows():
        document = row[text_col_name]
        try:
            output_list.append((index,) + document_relevance(query, document))
        except Exception as e:
            print(e)
        
    output_df = pd.DataFrame(
        output_list, columns=["index", "query", text_col_name, "prediction", "logprobs"]
    ).set_index('index')
    # Use exp() to convert logprobs into probability
    output_df["probability"] = output_df["logprobs"].apply(exp)
    # Add a new column for the model's confidence in its prediction
    output_df["confidence"] = output_df["logprobs"].apply(abs)
    # Reorder based on likelihood of being Yes
    output_df["yes_probability"] = output_df.apply(
        lambda x: x["probability"] * -1 + 1
        if x["prediction"] == "No"
        else x["probability"],
        axis=1,
    )

    # Return reranked results
    reranked_df = df.join(output_df, rsuffix='_reranked')
    df_out = reranked_df.sort_values(by=["yes_probability"], ascending=False)
    return df_out
