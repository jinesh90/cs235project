"""
Tagging sentiment on untagged twitter data.Input files are twitter data files from s3 location.output file
in .csv format with sentiment tag.
"""

import torch
from model.models import SentimentAnalyzerWithDropout
from config.parameters import DEVICE, SENTENCE_LENGTH, tokenizer
import pandas as pd
import time


def tag_twitter_sentiment(x):
    sentiment_class = ['negative', 'neutral', 'positive']
    model = SentimentAnalyzerWithDropout()
    model.load_state_dict(torch.load('../bin/dropout_sentiment.bin', map_location=torch.device('cpu')))
    model = model.eval()
    if isinstance(x, str):
        encoded_review = tokenizer.encode_plus(
            x,
            max_length=SENTENCE_LENGTH,
            add_special_tokens=True,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        model = model.to(DEVICE)
        input_ids = encoded_review['input_ids'].to(DEVICE)
        mask = encoded_review['attention_mask'].to(DEVICE)
        output = model(input_ids, mask)
        op1, prediction = torch.max(output, dim=1)
        print(sentiment_class[prediction])
        return sentiment_class[prediction]
    else:
        print("this text {} has issue".format(x))
        return "NaN"


t1 = time.time()
df = pd.read_excel("data/2020-01-clean.xlsx")
df["sentiment"] = df['clean_text'].apply(tag_twitter_sentiment)
t2 = time.time()
df.to_csv('data/2020-01-tagged.csv')


print("Time taken for tag operation is :{} seconds".format(t2-t1))