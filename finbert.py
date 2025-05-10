import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import yfinance as yf

os.environ["HF_HOME"] = "D:/programs/hf" # Sparen : )
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
finbert = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
text = "Google Stock Rises.lt RebuffsApple's Claim Al ls Hurting Search"

result = finbert(text)
print(result)