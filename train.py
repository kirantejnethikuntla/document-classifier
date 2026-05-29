import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

print("Loading dataset...")

# Load dataset
data = pd.read_csv("data/spam_updated.csv", encoding="latin-1")

# Keep only first two columns
data = data.iloc[:, :2]

# Rename columns
data.columns = ["label", "message"]

# Convert labels
data["label"] = data["label"].replace({
    "ham": "not spam",
    "spam": "spam"
})

print("\nDataset Shape:")
print(data.shape)

print("\nClass Distribution:")
print(data["label"].value_counts())

# Features and Labels
X = data["message"]
y = data["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create ML Pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])

print("\nTraining model...")

# Train model
model.fit(X_train, y_train)

# Test model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", accuracy)

# Create model folder if missing
os.makedirs("model", exist_ok=True)

# Save model
with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel saved successfully!")
print("Saved as: model/model.pkl")
