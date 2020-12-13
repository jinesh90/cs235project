"""
command line tool for detect sentiment of statement
USE
python cmd.py -t "I love pizza"

"""
import argparse
import torch
from model.models import SentimentAnalyzerWithDropout
from config.parameters import DEVICE, SENTENCE_LENGTH, tokenizer


parser = argparse.ArgumentParser(description='Detect sentiment from raw text')
parser.add_argument("--text", "--t", "-t","-text",type=str, required=True, help='raw text where find sentiment.')


args = parser.parse_args()

def main():
    if args.text:
        target_txt = str(args.text)
        sentiment_class = ['negative', 'neutral', 'positive']
        model = SentimentAnalyzerWithDropout()
        model.load_state_dict(torch.load('../bin/dropout_sentiment.bin', map_location=torch.device('cpu')))
        model = model.eval()
        if isinstance(target_txt, str):
            encoded_review = tokenizer.encode_plus(
                target_txt,
                max_length=SENTENCE_LENGTH,
                add_special_tokens=True,
                return_token_type_ids=False,
                pad_to_max_length=True,
                return_attention_mask=True,
                return_tensors='pt',
            )

            input_ids = encoded_review['input_ids'].to(DEVICE)
            attention_mask = encoded_review['attention_mask'].to(DEVICE)
            output = model(input_ids, attention_mask)
            op,prediction = torch.max(output, dim=1)
            print(sentiment_class[prediction])
        else:
            print("text has some issue")

main()