import os
import pandas as pd
import torch
import random
import logging
import evaluate
from datasets import Dataset as ds
from transformers import (
    BertForSequenceClassification,
    BertTokenizerFast,
    Trainer,
    TrainingArguments
)

logging.basicConfig(level=logging.INFO)


def readline_csv(split: float = 0.1, CSV_PATH: str = "data/sf.csv"):
    df = pd.read_csv(CSV_PATH)
    if "text" not in df.columns or "label" not in df.columns:
        logging.error("CSV file must contain 'text' and 'label' columns. Have the paralising file made a mistake?")
    df = df.sample(frac=1, random_state=random.randint(0, 10000)).reset_index(drop=True)
    val_size = int(len(df) * split)
    df_val = df.iloc[:val_size].reset_index(drop=True)
    df_train = df.iloc[val_size:].reset_index(drop=True)
    ds_train = ds.from_pandas(df_train)
    ds_val = ds.from_pandas(df_val)
    return {"train": ds_train, "validation": ds_val}

def dataset_tokenize(dataset_dict: dict, tokenizer: BertTokenizerFast, max_len: int = 256):
    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_len
        )
    tokenized = {}
    for split in ["train", "validation"]:
        ds = dataset_dict[split] # This ds is not that ds. IM lazy
        ds = ds.rename_column("label", "labels")
        ds = ds.map(tokenize_batch, batched=True, remove_columns=["text"])
        ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        tokenized[split] = ds
    return tokenized

def trainiere_finbert(
    model_pfad: str,
    tokenizer: BertTokenizerFast,
    datasets_tokenized: dict,
    output_dir: str = "finbert/finbert_ft",
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5
):
    model = BertForSequenceClassification.from_pretrained(model_pfad, num_labels=3)
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=learning_rate,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True
    )
    metric_acc = evaluate.load("accuracy")
    metric_f1 = evaluate.load("f1")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = torch.argmax(torch.tensor(logits), dim=-1).numpy()
        labels = labels
        acc = metric_acc.compute(predictions=preds, references=labels)["accuracy"]
        f1 = metric_f1.compute(predictions=preds, references=labels, average="weighted")["f1"]
        return {"accuracy": acc, "f1": f1}
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=datasets_tokenized["train"],
        eval_dataset=datasets_tokenized["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logging.info(f"Model and tokenizer saved in: {output_dir}")


def main():
    # --------------------------CONFIG-------------------------- #
    model = "finbert"
    ordner = "finbert/finbert_ft"
    epochen = 3
    batch = 16
    lr = 2e-5
    max_len_tokens = 256
    CSV_PATH = "data/sf.csv"

    logging.info("Loading tokenizer...")
    tokenizer = BertTokenizerFast.from_pretrained(model)
    ds_dict = readline_csv(split=0.1, CSV_PATH=CSV_PATH)
    ds_token = dataset_tokenize(ds_dict, tokenizer, max_len=max_len_tokens)

    logging.info("Starting Fine-Tuning...")
    trainiere_finbert(
        model_pfad=model,
        tokenizer=tokenizer,
        datasets_tokenized=ds_token,
        output_dir=ordner,
        num_epochs=epochen,
        batch_size=batch,
        learning_rate=lr
    )
