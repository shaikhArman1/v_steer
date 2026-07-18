import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# -------------------------------------------------
# Load Dataset
# -------------------------------------------------

print("Loading dataset...")

df = pd.read_csv("left_dataset.csv")

# -------------------------------------------------
# Separate Features and Labels
# -------------------------------------------------

X = df.drop("label", axis=1)
y = df["label"]

# -------------------------------------------------
# Split Dataset
# -------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Training Samples : {len(X_train)}")
print(f"Testing Samples  : {len(X_test)}")

# -------------------------------------------------
# Create Model
# -------------------------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

# -------------------------------------------------
# Train Model
# -------------------------------------------------

print("\nTraining model...\n")

model.fit(X_train, y_train)

# -------------------------------------------------
# Evaluate
# -------------------------------------------------

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("=" * 50)
print("Accuracy")
print("=" * 50)

print(f"{accuracy * 100:.2f}%")

print("\n")

print("=" * 50)
print("Confusion Matrix")
print("=" * 50)

print(confusion_matrix(y_test, predictions))

print("\n")

print("=" * 50)
print("Classification Report")
print("=" * 50)

print(classification_report(y_test, predictions))

# -------------------------------------------------
# Save Model
# -------------------------------------------------

joblib.dump(model, "left_model.pkl")

print("\nModel saved as left_model.pkl")
