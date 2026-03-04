import torch
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix

#load the IMDb dataset
dataset=load_dataset("imdb")

#load tokenizer
tokenizer=BertTokenizer.from_pretrained("bert-base-uncased")

#tokenize function
def tokenize_function(example):
    return tokenizer(example["text"],padding="max_length", truncation=True, max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

#remove original text and set format for pytorch
tokenized_datasets= tokenized_datasets.remove_columns(["text"])
tokenized_datasets.set_format("torch")

#load bert model
model= BertForSequenceClassification.from_pretrained("bert-base-uncased",num_labels=2)

#define metrics
def compute_metrics(pred):
    labels=pred.label_ids
    preds=pred.predictions.argmax(-1)

    precision, recall, f1, _=precision_recall_fscore_support(labels,preds,average="binary")
    
    acc=accuracy_score(labels, preds)

    cm= confusion_matrix(labels,preds)
    print("\nConfusion Matrix: ")
    print(cm)
    return{"accuracy":acc, "f1":f1, "precision":precision, "recall":recall}

# 6️⃣ Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=50,
)

# 7️⃣ Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"].shuffle(seed=42).select(range(500)),# small subset
    eval_dataset=tokenized_datasets["test"].shuffle(seed=42).select(range(500)), # small subset
    compute_metrics=compute_metrics,
)

# 8️⃣ Start training
trainer.train()

trainer.save_model("./model")
tokenizer.save_pretrained("./model")

