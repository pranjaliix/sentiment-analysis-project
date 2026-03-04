from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np

# 1️⃣ Load dataset
print("Loading dataset...")
dataset = load_dataset("imdb")
dataset=dataset.shuffle(seed=42)

train_texts = dataset["train"]["text"][:500]
train_labels = dataset["train"]["label"][:500]

test_texts = dataset["test"]["text"][:500]
test_labels = dataset["test"]["label"][:500]

# 2️⃣ Convert text → TF-IDF
print("Vectorizing text...")
vectorizer = TfidfVectorizer(
    max_features=10000,
    stop_words="english"
)

X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

# 3️⃣ Train Logistic Regression
print("Training Logistic Regression...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, train_labels)

# 4️⃣ Predictions
print("Evaluating...")
predictions = model.predict(X_test)

# 5️⃣ Metrics
accuracy = accuracy_score(test_labels, predictions)
precision, recall, f1, _ = precision_recall_fscore_support(
    test_labels,
    predictions,
    average="binary"
)

cm = confusion_matrix(test_labels, predictions)

print("\n=== Logistic Regression Baseline Results ===")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)