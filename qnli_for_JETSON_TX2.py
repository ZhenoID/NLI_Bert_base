from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
import torch
import numpy as np

dataset = load_dataset("glue", "qnli")
print(dataset["train"][0])

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_fn(batch):
    return tokenizer(
        batch["question"],
        batch["sentence"],
        truncation=True,
        max_length=128
    )

tokenized = dataset.map(tokenize_fn, batched=True)
tokenized = tokenized.remove_columns(["question", "sentence", "idx"])
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type != "cuda":
    raise RuntimeError("CUDA GPU is not available. Stop the process.")

print("CUDA available:", torch.cuda.is_available())
print("Using device:", device)
print("GPU name:", torch.cuda.get_device_name(0))

model.to(device)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    accuracy = (preds == labels).mean()

    return {
        "accuracy": float(accuracy)
    }

train = tokenized["train"]
val = tokenized["validation"]

training_args = TrainingArguments(
    output_dir="./bert_qnli_fulldataset_1epoch",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=1000,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=1,
    learning_rate=4e-5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train,
    eval_dataset=val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()