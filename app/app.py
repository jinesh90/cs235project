"""
web application for quick sentiment test. run as standalone web server.
"""
import torch
from flask import Flask, request, render_template
from model.models import SentimentAnalyzerWithDropout
from config.parameters import DEVICE, SENTENCE_LENGTH,tokenizer


app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('template.html')

@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text']
    sentiment_class = ['negative', 'neutral', 'positive']
    model = SentimentAnalyzerWithDropout()
    model.load_state_dict(torch.load('../bin/dropout_sentiment.bin', map_location=torch.device('cpu')))
    model = model.eval()
    if isinstance(text, str):
        encoded_review = tokenizer.encode_plus(
            text,
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
        _, prediction = torch.max(output, dim=1)
        print(sentiment_class[prediction])

    return sentiment_class[prediction]


if __name__ == '__main__':
    app.run()
