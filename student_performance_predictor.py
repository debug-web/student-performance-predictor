#  Student Performance Predictor
#  Course: Fundamentals of AI and ML – BYOP Capstone Project
#  Model  : Decision Tree Classifier (Supervised Learning)
#  Dataset: UCI Student Performance (student-mat.csv)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import LabelEncoder

print("=" * 60)
print("  Student Performance Predictor")
print("=" * 60)

try:
    df = pd.read_csv("student-mat.csv", sep=";")
    print(f"\n[✓] Dataset loaded — {df.shape[0]} students, {df.shape[1]} columns\n")
except FileNotFoundError:
    raise SystemExit(
        "\n[✗] 'student-mat.csv' not found.\n"
        "    Download it from:\n"
        "    https://archive.ics.uci.edu/dataset/320/student+performance\n"
        "    and place it in the same directory as this script."
    )

print("--- Dataset Overview ---")
print(df[["age", "studytime", "failures", "absences", "G1", "G2", "G3"]].describe().round(2))

df["pass"] = (df["G3"] >= 10).astype(int)

pass_rate = df["pass"].mean() * 100
print(f"\n[i] Pass rate in dataset: {pass_rate:.1f}%")

binary_cols = ["school", "sex", "address", "famsize", "Pstatus",
               "schoolsup", "famsup", "paid", "activities",
               "nursery", "higher", "internet", "romantic"]
le = LabelEncoder()
for col in binary_cols:
    df[col] = le.fit_transform(df[col])

FEATURES = [
    "age", "studytime", "failures", "absences",
    "G1", "G2",                          # previous period grades
    "famrel", "freetime", "goout",       # lifestyle
    "Dalc", "Walc", "health",            # health & habits
    "higher", "internet", "paid",        # support factors
]

X = df[FEATURES]
y = df["pass"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n[✓] Train size: {len(X_train)} | Test size: {len(X_test)}")

model = DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, random_state=42)
model.fit(X_train, y_train)
print("\n[✓] Decision Tree trained successfully.")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n--- Model Performance ---")
print(f"Accuracy : {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Fail", "Pass"]))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Student Performance Predictor — Results", fontsize=14, fontweight="bold")

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fail", "Pass"])
disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
axes[0].set_title("Confusion Matrix")

importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
importances.plot(kind="barh", ax=axes[1], color="steelblue", edgecolor="white")
axes[1].set_title("Feature Importances")
axes[1].set_xlabel("Importance Score")
axes[1].axvline(0, color="black", linewidth=0.5)

plt.tight_layout()
plt.savefig("results.png", dpi=150, bbox_inches="tight")
print("\n[✓] Results plot saved → results.png")
plt.show()

print("\n--- Top Decision Rules (depth ≤ 3) ---")
rules = export_text(model, feature_names=FEATURES, max_depth=3)
print(rules)

print("\n--- Quick Prediction Demo ---")
sample = pd.DataFrame([{
    "age": 17, "studytime": 2, "failures": 0, "absences": 4,
    "G1": 12, "G2": 13,
    "famrel": 4, "freetime": 3, "goout": 2,
    "Dalc": 1, "Walc": 1, "health": 4,
    "higher": 1, "internet": 1, "paid": 0
}])

prediction = model.predict(sample)[0]
probability = model.predict_proba(sample)[0]
label = "PASS ✓" if prediction == 1 else "FAIL ✗"
print(f"Sample student prediction : {label}")
print(f"  Fail probability : {probability[0]*100:.1f}%")
print(f"  Pass probability : {probability[1]*100:.1f}%")

