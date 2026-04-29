import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load saved model
MODEL_NAME = "Pranjalii/sentiment-analysis-bert"

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained(MODEL_NAME)

model.eval()

def predict_sentiment(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()

    return "Positive 😊" if prediction == 1 else "Negative 😔"

# Test loop
# while True:
#     text = input("\nEnter a sentence (or type 'exit'): ")
#     if text.lower() == "exit":
#         break
#     print("Prediction:", predict_sentiment(text))/

