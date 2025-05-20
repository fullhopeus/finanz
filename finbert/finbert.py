import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import yfinance as yf
from datasets import load_dataset

os.environ["HF_HOME"] = "D:/programs/hf" # Sparen : )
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
dataset = load_dataset('csv', data_files='data/sf.csv')
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
