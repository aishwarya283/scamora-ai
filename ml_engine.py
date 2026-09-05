# ============================================================
# SCAMORA AI - MACHINE LEARNING ENGINE
# Text Classification + Keyword Risk + AASIST Deepfake
# ============================================================

import os
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from deepfake_detector import detect_deepfake


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = (
    BASE_DIR.parent
    / "test_samples"
    / "text"
    / "BETTER30_cleaned.csv"
)

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "scamora_model.pkl"
TFIDF_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"


# ============================================================
# 2. KEYWORD RISK SYSTEM
# ============================================================

HIGH_RISK_KEYWORDS = {
    "otp": 25,
    "one time password": 25,
    "cvv": 25,
    "password": 20,
    "digital arrest": 30,
    "account blocked": 25,
    "account will be blocked": 25,
    "transfer money": 25,
    "send money": 25,
    "police": 15,
    "arrest": 25,
    "legal action": 20,
    "click this link": 20,
    "share your details": 20,
    "verify your account": 15,
    "verify immediately": 20,
    "customs": 15,
    "upi": 15,
    "credit card": 15,
    "debit card": 15,
    "urgent": 10,
    "immediately": 10,
    "bank": 5,
    "payment": 5,
    "refund": 10,
    "fine": 10,
}


def calculate_keyword_risk(text):
    """
    Calculate scam risk using weighted scam keywords.
    """

    text = str(text).lower()

    matched_keywords = []
    risk_score = 0

    for keyword, weight in HIGH_RISK_KEYWORDS.items():
        if keyword in text:
            matched_keywords.append(keyword)
            risk_score += weight

    risk_score = min(risk_score, 100)

    return risk_score, matched_keywords


# ============================================================
# 3. LOAD DATASET
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# 4. FIND TEXT COLUMN
# ============================================================

possible_text_columns = [
    "CONVERSATION_STEP_TEXT",
    "TEXT",
    "text",
    "Text",
    "CONVERSATION_TEXT",
]

text_column = None

for column in possible_text_columns:
    if column in df.columns:
        text_column = column
        break

if text_column is None:
    raise ValueError(
        "Text column not found. Available columns are: "
        + str(df.columns.tolist())
    )


# ============================================================
# 5. FIND LABEL COLUMN
# ============================================================

possible_label_columns = [
    "LABEL",
    "label",
    "Label",
]

label_column = None

for column in possible_label_columns:
    if column in df.columns:
        label_column = column
        break

if label_column is None:
    raise ValueError(
        "LABEL column not found. Available columns are: "
        + str(df.columns.tolist())
    )


print("\nText column:", text_column)
print("Label column:", label_column)


# ============================================================
# 6. CLEAN DATA
# ============================================================

df = df.dropna(subset=[text_column, label_column])

df[text_column] = df[text_column].astype(str)
df[label_column] = df[label_column].astype(str).str.strip()

print("\nDataset after removing missing values:")
print(df.shape)


# ============================================================
# 7. ORIGINAL LABEL DISTRIBUTION
# ============================================================

print("\nLabel distribution:")
print(df[label_column].value_counts())


# ============================================================
# 8. MAP LABELS TO SCAMORA CATEGORIES
# ============================================================

def map_label(label):
    """
    Convert original dataset labels into:

    NORMAL
    SUSPICIOUS
    SCAM
    """

    label = str(label).strip().lower()

    if label in [
        "scam",
        "scam_response",
        "potential_scam",
    ]:
        return "SCAM"

    if label in [
        "suspicious",
        "slightly_suspicious",
        "highly_suspicious",
    ]:
        return "SUSPICIOUS"

    if label in [
        "neutral",
        "legitimate",
    ]:
        return "NORMAL"

    return None


X = df[text_column]
y = df[label_column].apply(map_label)


# ============================================================
# 9. REMOVE UNKNOWN LABELS
# ============================================================

valid_rows = y.notna()

X = X[valid_rows]
y = y[valid_rows]

print("\nScamora AI label distribution:")
print(y.value_counts())

print("\nNumber of text samples:", len(X))
print("Number of classes:", y.nunique())


# ============================================================
# 10. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 11. TF-IDF
# ============================================================

print("\nCreating TF-IDF features...")

tfidf = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    max_features=10000,
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("TF-IDF training shape:", X_train_tfidf.shape)
print("TF-IDF testing shape:", X_test_tfidf.shape)


# ============================================================
# 12. TRAIN MODEL
# ============================================================

print("\nTraining Logistic Regression model...")

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
)

model.fit(X_train_tfidf, y_train)

print("Model training completed successfully!")


# ============================================================
# 13. MODEL EVALUATION
# ============================================================

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0,
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0,
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0,
)


print("\n======================================")
print("       MODEL EVALUATION")
print("======================================")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")


# ============================================================
# 14. CLASSIFICATION REPORT
# ============================================================

print("\n======================================")
print("       CLASSIFICATION REPORT")
print("======================================")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0,
    )
)


# ============================================================
# 15. CONFUSION MATRIX
# ============================================================

print("\n======================================")
print("       CONFUSION MATRIX")
print("======================================")

cm = confusion_matrix(y_test, y_pred)

print(cm)


# ============================================================
# 16. SAVE MODEL
# ============================================================

joblib.dump(model, MODEL_PATH)
joblib.dump(tfidf, TFIDF_PATH)

print("\n======================================")
print("       MODEL SAVED SUCCESSFULLY")
print("======================================")

print("Model saved to:", MODEL_PATH)
print("TF-IDF saved to:", TFIDF_PATH)


# ============================================================
# 17. TEXT PREDICTION
# ============================================================

def predict_text(text):
    """
    Analyze text using:

    1. Logistic Regression
    2. Scam keyword risk
    3. Combined risk score

    Returns a dictionary containing all results.
    """

    text = str(text).strip()

    if not text:
        raise ValueError("Text cannot be empty.")

    # ML prediction
    text_features = tfidf.transform([text])

    ml_prediction = model.predict(text_features)[0]

    probabilities = model.predict_proba(text_features)[0]

    class_probabilities = dict(
        zip(model.classes_, probabilities)
    )

    ml_scam_probability = class_probabilities.get(
        "SCAM",
        0.0,
    )

    # Keyword analysis
    keyword_risk, matched_keywords = calculate_keyword_risk(
        text
    )

    # Combined risk
    combined_risk = (
        (ml_scam_probability * 100 * 0.40)
        + (keyword_risk * 0.60)
    )

    combined_risk = min(
        max(combined_risk, 0),
        100,
    )

    # Final classification
    if keyword_risk >= 50:
        final_prediction = "SCAM"

    elif combined_risk >= 50:
        final_prediction = "SCAM"

    elif combined_risk >= 25:
        final_prediction = "SUSPICIOUS"

    else:
        final_prediction = "NORMAL"

    return {
        "prediction": final_prediction,
        "risk_score": round(combined_risk, 2),
        "ml_prediction": ml_prediction,
        "ml_scam_probability": round(
            float(ml_scam_probability),
            4,
        ),
        "keyword_risk": keyword_risk,
        "matched_keywords": matched_keywords,
        "class_probabilities": {
            key: round(float(value), 4)
            for key, value in class_probabilities.items()
        },
    }


# ============================================================
# 18. COMBINED AUDIO + TEXT ANALYSIS
# ============================================================

def analyze_call(audio_path, text):
    """
    Combine:

    Text scam detection
    +
    AASIST deepfake voice detection
    """

    # -------------------------
    # TEXT ANALYSIS
    # -------------------------

    text_result = predict_text(text)

    # -------------------------
    # AUDIO ANALYSIS
    # -------------------------

    audio_label, audio_confidence = detect_deepfake(
        audio_path
    )

    # -------------------------
    # FINAL DECISION
    # -------------------------

    if audio_label == "FAKE":

        final_result = "AI DEEPFAKE VOICE"

    elif text_result["prediction"] == "SCAM":

        final_result = "SCAM"

    elif text_result["prediction"] == "SUSPICIOUS":

        final_result = "SUSPICIOUS"

    else:

        final_result = "NORMAL"

    return {
        "final_result": final_result,

        "text_result": text_result["prediction"],

        "text_risk_score": text_result["risk_score"],

        "ml_scam_probability": (
            text_result["ml_scam_probability"]
        ),

        "keyword_risk": text_result["keyword_risk"],

        "matched_keywords": (
            text_result["matched_keywords"]
        ),

        "audio_result": audio_label,

        "audio_confidence": round(
            float(audio_confidence),
            4,
        ),
    }


# ============================================================
# 19. SAMPLE TEXT TEST
# ============================================================

if __name__ == "__main__":

    print("\n======================================")
    print("       SAMPLE TEXT TEST")
    print("======================================")

    samples = [
        (
            "SCAM",
            "Your bank account will be blocked. "
            "Give me your OTP immediately."
        ),
        (
            "NORMAL",
            "Your food delivery will arrive "
            "in 20 minutes."
        ),
        (
            "SUSPICIOUS",
            "Please verify your account "
            "using the link we sent you."
        ),
    ]

    for expected, text in samples:

        result = predict_text(text)

        print("\nExpected:", expected)
        print("Text:", text)
        print("Prediction:", result["prediction"])
        print("Risk:", result["risk_score"])
        print(
            "ML Scam Probability:",
            result["ml_scam_probability"],
        )
        print(
            "Keyword Risk:",
            result["keyword_risk"],
        )
        print(
            "Matched Keywords:",
            result["matched_keywords"],
        )

    print("\n======================================")
    print("       SCAMORA ML ENGINE READY")
    print("======================================")