"""
various deep neural networks define here, all models are derived from torch nn module
"""
import torch.nn as nn
from config.parameters import bert


class SimpleSentimentAnalyzer(nn.Module):
    """
    simplest sentiment analyzer model
    """
    def __init__(self, sentiments=3):
        super(SimpleSentimentAnalyzer, self).__init__()
        self.model = bert
        self.output = nn.Linear(self.model.config.hidden_size, sentiments)

    def forward(self, input_ids, mask):
        op1, op2 = self.model(input_ids=input_ids, attention_mask=mask)
        return self.output(op2)


class SentimentAnalyzerWithDropout(nn.Module):
    """
    deep neural network with 35% drop out.
    """

    def __init__(self, sentiments=3):
        super(SentimentAnalyzerWithDropout, self).__init__()
        self.model = bert
        self.dropout = nn.Dropout(p=0.35)
        self.output = nn.Linear(self.model.config.hidden_size, sentiments)

    def forward(self, input_ids, mask):
        op1, op2 = self.model(input_ids=input_ids, attention_mask=mask)
        op3 = self.dropout(op2)
        return self.output(op3)


class DeepSentimentAnalyzer(nn.Module):
    """
    deeper neural network with dropout
    """
    def __init__(self, sentiments=3):
        super(DeepSentimentAnalyzer, self).__init__()
        self.model = bert
        self.dropout1 = nn.Dropout(p=0.35)
        self.deeplayer = nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)
        self.dropout2 = nn.Dropout(p=0.35)
        self.output = nn.Linear(self.model.config.hidden_size, sentiments)

    def forward(self, input_ids, mask):
        op1, op2 = self.model(input_ids=input_ids, attention_mask=mask)
        op3 = self.dropout1(op2)
        op4 = self.deeplayer(op3)
        op5 = self.dropout2(op4)
        return self.output(op5)