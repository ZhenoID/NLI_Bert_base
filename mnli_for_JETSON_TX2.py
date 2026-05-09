from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import TrainingArguments
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification
from transformers import Trainer

dataset = load_dataset("glue", "mnli")
print(dataset["train"][0])
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_fn(batch):
    return tokenizer(
        batch["premise"],
        batch["hypothesis"],
        truncation = True,
        padding = "max_length",
        max_length = 128
    )
tokenized = dataset.map(tokenize_fn, batched = True)
print(tokenized["train"][0].keys())
print(tokenized["train"][0])
tokenized = tokenized.remove_columns(["premise", "hypothesis", "idx"])
tokenized.set_format("torch")
print(tokenized["train"][0].keys())
print(tokenized["train"][0])

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

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
val = tokenized["validation_matched"]


training_args = TrainingArguments(
    output_dir="./bert_mnli_1epoch",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=1000,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
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
    compute_metrics=compute_metrics,
)

trainer.train()

matched_results = trainer.evaluate(eval_dataset=tokenized["validation_matched"])
print("Matched validation:", matched_results)

mismatched_results = trainer.evaluate(eval_dataset=tokenized["validation_mismatched"])
print("Mismatched validation:", mismatched_results)