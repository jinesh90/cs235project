import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
from model.models import DeepSentimentAnalyzer,SimpleSentimentAnalyzer,SentimentAnalyzerWithDropout
from dataprocessor.preprocessor import GetLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, roc_auc_score,accuracy_score
from config.parameters import RANDOM_SEED, training_test_size, validation_test_size,tokenizer


class ModelEvaluation:
    """
    model evaluation on test data and get confusion matrix.
    """
    def __init__(self, model, dataframe, device="cpu", model_name="model.bin"):
        """

        :param model:
        :param dataframe:
        :param device:
        :param model_name:
        """
        self.df = dataframe
        self.model = model
        df_train, df_test = train_test_split(self.df, test_size=training_test_size, random_state=RANDOM_SEED)
        df_val, df_test = train_test_split(df_test, test_size=validation_test_size, random_state=RANDOM_SEED)
        test_loader = GetLoader(df_test, tokenizer)
        self.loader = test_loader.get()
        self.total_loss = []
        self.correct = 0
        self.device = device
        self.sentiment_class = ["negative", "neutral", "positive"]
        self.model_name = model_name
        self.frame_length = len(df_test)

    def evaluation(self, loss_function):
        """

        :param loss_function:
        :return:
        """
        total_loss = []
        self.model = self.model.eval()
        test_accuracy = 0
        with torch.no_grad():
            for data in self.loader:
                input_ids = data["input_ids"].to(self.device)
                attention_mask = data["attention_mask"].to(self.device)
                sentiment = data["sentiment"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    mask=attention_mask
                )
                _, preds = torch.max(outputs, dim=1)
                loss = loss_function(outputs, sentiment)

                self.correct += torch.sum(torch.tensor(preds == sentiment))
                total_loss.append(loss.item())
                test_accuracy = self.correct.double() / self.frame_length

        return test_accuracy

    def get_confusion_matrix(self):
        """
        get confusion matrix from data
        :return:
        """
        self.model = self.model.eval()
        sentiments = []
        model_pred = []
        pred_porb = []
        actual_sentiment = []
        with torch.no_grad():
            for data in self.loader:
                texts = data["text"]
                input_ids = data["input_ids"].to(self.device)
                attention_mask = data["attention_mask"].to(self.device)
                sentiment = data["sentiment"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    mask=attention_mask
                )
                _, preds = torch.max(outputs, dim=1)

                probs = F.softmax(outputs, dim=1)
                sentiments.extend(texts)
                model_pred.extend(preds)
                pred_porb.extend(probs)
                actual_sentiment.extend(sentiment)

        model_pred = torch.stack(model_pred).cpu()
        actual_sentiment = torch.stack(actual_sentiment).cpu()


        cm = confusion_matrix(actual_sentiment, model_pred)
        print (cm)
        hmap = sns.heatmap(pd.DataFrame(cm, index=self.sentiment_class, columns=self.sentiment_class), annot=True, fmt="d")
        hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
        hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=20, ha='right')
        plt.ylabel('True sentiment')
        plt.xlabel('Predicted sentiment')
        plt.savefig("{}.png".format(self.model_name))


def main():
    model = DeepSentimentAnalyzer()
    model.load_state_dict(torch.load('../bin/deep_sentiment_with_bert_uncased.bin', map_location=torch.device('cpu')))
    model = model.eval()
    loss_function = nn.CrossEntropyLoss().to('cpu')
    df = pd.read_csv('../data/data.csv')
    me = ModelEvaluation(model, df, model_name='deep_sentiment_with_bert_uncased')
    #print(me.evaluation(loss_function))
    me.get_confusion_matrix()

main()