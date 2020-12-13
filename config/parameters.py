"""
model constant and hyper tuning model hypertuning params
"""
import torch
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup, get_constant_schedule, AdamWeightDecay


# Set device for running model, in case of GPU ,it will automatically use GPU.
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#DEVICE = "cpu"

# BERT pre trained model, which defined tokens, buffer etc.
PRE_TRAINED_MODEL = "bert-base-cased"

# Define BERT based tokenizer here.
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL)

# Define BERT model
bert = BertModel.from_pretrained(PRE_TRAINED_MODEL)

# Maximum input tensor length based on sentence.
SENTENCE_LENGTH = 192

# Random seed for model
RANDOM_SEED = 35

# Training test size 10%
training_test_size = 0.1

# Validation test size 50%
validation_test_size = 0.5

# Worker for data loader
WORKER = 4

# Batch size
BATCH = 16

# Epoch for training
EPOCHS = 10

# Optimizer parameters for fine tune BERT.
LEARNING_RATE = 1e-5

# Optimizer EPS
EPS = 1e-6

# Correct bias
CORRECT_BIAS = False

# optimizer class
adam = AdamW

# schedule
scheduler = get_linear_schedule_with_warmup

# model save path
SAVE_PATH = 'bin/'

# dataframe path
DATAFRAME = 'data/data.csv'