"""
data preprocessor step.
1) taking input raw text
2) tokenize it. i.e adding special BERT classification token and buffer tokens
3) convert text into special BERT requirements , input_ids and attention_mask
4) returning object that has preprocessed data.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from config.parameters import SENTENCE_LENGTH, BATCH, WORKER


class PreProcessor(Dataset):
    """
    pre-processing text, returning object with attention mask and encoded text
    """

    def __init__(self, text, sentiment, tokenizer):
        self.text = text
        self.sentiment = sentiment
        self.tokenizer = tokenizer

    def __str__(self):
        return ""

    def __len__(self):
        if self.text is not None:
            return len(self.text)

    def __getitem__(self, item):
        if self.text[item] is not None:
            encoded_string = self.tokenizer.encode_plus(self.text[item],
                                                         max_length=SENTENCE_LENGTH,
                                                         return_attention_mask=True,
                                                         pad_to_max_length=True,
                                                         add_special_tokens=True,
                                                         return_tensors='pt')
            preprocessed = {
                "text": self.text[item],
                "sentiment": torch.tensor(self.sentiment[item], dtype=torch.long),
                "input_ids": encoded_string["input_ids"].flatten(),
                "attention_mask": encoded_string["attention_mask"].flatten()
            }

            return preprocessed
        else:
            print("*"*100)
            print("some text has issue")
            print("*"*100)


class GetLoader:
    """
    data loader class
    """
    def __init__(self, dataframe, tokenizer):
        self.df = dataframe
        self.tokenizer = tokenizer

    def get(self):
        return DataLoader(PreProcessor(text=self.df.statement.to_numpy(),
                                       sentiment=self.df.sentiment.to_numpy(),
                                       tokenizer=self.tokenizer,
                                       ),
                          batch_size=BATCH,
                          num_workers=WORKER)





# df = pd.read_csv('../data/data.csv')
# print(df.head())
#
# tl = GetLoader(df, tokenizer)
# train_loader = tl.get()
