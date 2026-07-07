import pandas as pd

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)

MODEL_NAME = "google/mt5-small"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

train_df = pd.read_csv("data/processed/train.csv")
val_df = pd.read_csv("data/processed/val.csv")

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

def preprocess(examples):

    model_inputs = tokenizer(
        examples["source"],
        truncation=True,
        max_length=128
    )

    labels = tokenizer(
        examples["target"],
        truncation=True,
        max_length=128
    )

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

train_dataset = train_dataset.map(
    preprocess,
    batched=True,
    remove_columns=train_dataset.column_names
)

val_dataset = val_dataset.map(
    preprocess,
    batched=True,
    remove_columns=val_dataset.column_names
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model
)

training_args = Seq2SeqTrainingArguments(
    output_dir="models/oku_model",

    num_train_epochs=20,

    learning_rate=5e-5,

    per_device_train_batch_size=2,

    per_device_eval_batch_size=2,

    weight_decay=0.01,

    eval_strategy="epoch",

    save_strategy="epoch",

    load_best_model_at_end=True,

    metric_for_best_model="eval_loss",

    predict_with_generate=True,

    logging_steps=20,

    save_total_limit=3,

    fp16=False
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=tokenizer,
    data_collator=data_collator
)

trainer.train()

trainer.save_model("models/oku_model")
tokenizer.save_pretrained("models/oku_model")

print("Training complete")