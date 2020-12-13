"""
main file to traine and save model, before running this, if you want to tune parameters, Please change in
config/parameters
"""

import pandas as pd
import torch.nn as nn
from builder import ModelBuilder
from model.models import SimpleSentimentAnalyzer, SentimentAnalyzerWithDropout, DeepSentimentAnalyzer
from dataprocessor.preprocessor import GetLoader
from sklearn.model_selection import train_test_split
from config.parameters import tokenizer, adam, scheduler, RANDOM_SEED, training_test_size, validation_test_size
from config.parameters import LEARNING_RATE, CORRECT_BIAS
from config.parameters import EPOCHS, DEVICE, DATAFRAME


def main():

    # read training data from data/data.csv.
    df = pd.read_csv(DATAFRAME)

    # model name
    model_name = "model_test"

    # split  data into train,validation, test
    df_train, df_test = train_test_split(df, test_size=training_test_size, random_state=RANDOM_SEED)
    df_val, df_test = train_test_split(df_test, test_size=validation_test_size, random_state=RANDOM_SEED)

    tl = GetLoader(df_train, tokenizer)
    train_loader = tl.get()

    tsl = GetLoader(df_test, tokenizer)


    tv = GetLoader(df_val, tokenizer)
    val_loader = tv.get()
    #
    model = SentimentAnalyzerWithDropout()
    model = model.to(DEVICE)

    total_steps = len(train_loader) * EPOCHS

    # choose optimizer.
    optimizer = adam(model.parameters(), lr=LEARNING_RATE, correct_bias=CORRECT_BIAS)
    model_schedular = scheduler(optimizer, num_training_steps=total_steps, num_warmup_steps=0)

    # select loss function, this is also a hyper params.
    loss_function = nn.CrossEntropyLoss().to(DEVICE)

    # build model.
    m1 = ModelBuilder(model=model, loss_function=loss_function, optimizer=optimizer,
                      model_scheduler=model_schedular, model_name="{}.bin".format(model_name))
    # start training.
    m1.start(train_loader, val_loader, len(df_train), len(df_val))


main()
