"""
Heart Disease Predictor — Training Script
Dataset: heart.csv (UCI Cleveland, 1025 samples, 13 features)
Run: python train_model.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_auc_score)
import joblib
import json
import os

# ── Load Data ──────────────────────────────────────────────────────────────
df = pd.read_csv("heart.csv")
print(f"Dataset shape: {df.shape}")
print(f"Target distribution:\n{df['target'].value_counts()}\n")

X = df.drop("target", axis=1)
y = df["target"]

# ── Train / Test Split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Scaler (for LR / SVM) ──────────────────────────────────────────────────
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ── Train All Models ───────────────────────────────────────────────────────
models = {
    "Random Forest":       (RandomForestClassifier(n_estimators=100, random_state=42), False),
    "Gradient Boosting":   (GradientBoostingClassifier(random_state=42), False),
    "Logistic Regression": (LogisticRegression(max_iter=1000, random_state=42), True),
    "SVM":                 (SVC(probability=True, random_state=42), True),
}

results = {}
print("=" * 50)
for name, (model, needs_scale) in models.items():
    Xtr = X_train_s if needs_scale else X_train
    Xte = X_test_s  if needs_scale else X_test
    model.fit(Xtr, y_train)
    preds = model.predict(Xte)
    acc   = accuracy_score(y_test, preds)
    auc   = roc_auc_score(y_test, model.predict_proba(Xte)[:, 1])
    cv    = cross_val_score(model, Xtr, y_train, cv=5, scoring="accuracy").mean()
    results[name] = {"model": model, "acc": acc, "auc": auc, "cv": cv, "scaled": needs_scale}
    print(f"{name:<25}  Acc={acc:.4f}  AUC={auc:.4f}  CV={cv:.4f}")

print("=" * 50)

# ── Best Model ─────────────────────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]["acc"])
best      = results[best_name]
print(f"\nBest model: {best_name}  (Accuracy: {best['acc']:.4f})\n")

print("Classification Report:")
Xte = X_test_s if best["scaled"] else X_test
print(classification_report(y_test, best["model"].predict(Xte),
                             target_names=["No Disease", "Disease"]))

print("Confusion Matrix:")
print(confusion_matrix(y_test, best["model"].predict(Xte)))

# ── Feature Importance ─────────────────────────────────────────────────────
if hasattr(best["model"], "feature_importances_"):
    fi = pd.Series(best["model"].feature_importances_, index=X.columns)
    print("\nFeature Importances:")
    print(fi.sort_values(ascending=False).to_string())

# ── Save Artifacts ─────────────────────────────────────────────────────────
os.makedirs("saved_model", exist_ok=True)
joblib.dump(best["model"], "saved_model/model.pkl")
joblib.dump(scaler,        "saved_model/scaler.pkl")

meta = {
    "best_model":   best_name,
    "accuracy":     best["acc"],
    "auc":          best["auc"],
    "cv_accuracy":  best["cv"],
    "needs_scaling": best["scaled"],
    "features":     X.columns.tolist(),
    "all_results":  {k: {"acc": v["acc"], "auc": v["auc"]} for k, v in results.items()},
}
with open("saved_model/meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("\nSaved to saved_model/  (model.pkl, scaler.pkl, meta.json)")

# ── Quick Predict Function ─────────────────────────────────────────────────
def predict_patient(age, sex, cp, trestbps, chol, fbs, restecg,
                    thalach, exang, oldpeak, slope, ca, thal):
    """Predict heart disease probability for a single patient."""
    sample = pd.DataFrame([{
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
        "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
        "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
    }])
    model = best["model"]
    X_in  = scaler.transform(sample) if best["scaled"] else sample
    prob  = model.predict_proba(X_in)[0][1]
    pred  = model.predict(X_in)[0]
    print(f"\nPrediction: {'HEART DISEASE' if pred == 1 else 'NO HEART DISEASE'}")
    print(f"Probability: {prob:.2%}")
    return pred, prob

# Example usage:
# predict_patient(age=63, sex=1, cp=3, trestbps=145, chol=233,
#                 fbs=1, restecg=0, thalach=150, exang=0,
#                 oldpeak=2.3, slope=0, ca=0, thal=1)
