from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd

def rerank_with_transformers(query: str, df: pd.DataFrame, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
    # Initialize the model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Prepare the data for the model
    data = [(query, text) for text in df['text'].tolist()]

    # Tokenize the data
    features = tokenizer(*zip(*data), padding=True, truncation=True, return_tensors="pt")

    # Predict the scores
    model.eval()
    with torch.no_grad():
        scores = model(**features).logits

    # Convert scores to numpy array
    scores = scores.detach().numpy()

    # Add the scores to the dataframe
    df['scores'] = scores

    # Sort the dataframe by the scores in descending order
    df = df.sort_values(by='scores', ascending=False)

    return df
