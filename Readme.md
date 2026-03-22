# 🫀 Heart Disease Predictor

A machine learning web app that predicts the likelihood of heart disease based on 13 clinical features — no server, no setup, just open the HTML file in your browser and it works.

Built using a **Random Forest classifier** trained on the UCI Cleveland Heart Disease dataset (1,025 patients). The entire model is embedded inside the HTML file as decision trees, so inference runs directly in JavaScript — completely offline.

---

## 🚀 Live Demo

Just download `heart_disease_predictor.html` and open it in any browser. That's it.

No Python. No Node. No internet required after download.

---

## 📸 Preview

> Fill in patient details using sliders and toggle buttons → click **Run Prediction** → get an instant risk assessment with a probability score, animated gauge, and feature importance breakdown.

---

## 📊 Model Performance

| Model | Test Accuracy | Notes |
|---|---|---|
| **Random Forest** ✅ | **96.10%** | Best performer — used in the app |
| Gradient Boosting | 93.17% | Strong but slightly weaker |
| SVM | 88.78% | Good, requires feature scaling |
| Logistic Regression | 79.51% | Simple baseline |

The Random Forest was trained with 20 trees (max depth 8) on 820 samples and evaluated on 205 held-out samples.

---

## 🧠 How It Works

The model is trained in Python using scikit-learn, then the decision trees are exported as a JSON structure and embedded directly in the HTML file. When you click "Run Prediction", the browser walks through each tree, averages the class probabilities, and shows you the result — all without any network call.

```
User Input → JavaScript RF Inference → Risk Probability → Visual Result
```

This means:
- Works 100% offline
- No data ever leaves your device
- Loads instantly — no API latency

---

## 📁 Project Structure

```
heart-disease-predictor/
│
├── heart_disease_predictor.html   # The full app — open this in your browser
├── train_model.py                 # Python script to retrain the model
├── heart.csv                      # Dataset (UCI Cleveland, 1025 samples)
├── model.pkl                      # Saved Random Forest model (joblib)
├── scaler.pkl                     # Fitted StandardScaler
├── trees.json                     # Exported RF trees (used by the HTML app)
└── meta.json                      # Model metadata and accuracy scores
```

---

## 🔬 Dataset & Features

**Source:** UCI Machine Learning Repository — Cleveland Heart Disease Dataset

**1,025 patients** | **13 input features** | **Binary target** (0 = No Disease, 1 = Disease)

| Feature | Description |
|---|---|
| `age` | Age in years |
| `sex` | Sex (0 = Female, 1 = Male) |
| `cp` | Chest pain type (0–3) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dL) |
| `fbs` | Fasting blood sugar > 120 mg/dL (1 = True) |
| `restecg` | Resting ECG results (0–2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina (0/1) |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of peak exercise ST segment (0–2) |
| `ca` | Number of major vessels colored by fluoroscopy (0–3) |
| `thal` | Thalassemia type (0–3) |

**Top predictors by feature importance:**
```
Major Vessels (ca)     ████████████████░░░░  18.1%
Thalassemia (thal)     ██████████████░░░░░░  16.2%
Chest Pain (cp)        ██████████░░░░░░░░░░  11.8%
Max Heart Rate         ████████░░░░░░░░░░░░   9.9%
ST Depression          ███████░░░░░░░░░░░░░   9.2%
```

---

## 🛠️ Retrain the Model Yourself

If you want to retrain from scratch or experiment with different parameters:

```bash
# 1. Install dependencies
pip install scikit-learn pandas numpy joblib

# 2. Make sure heart.csv is in the same folder, then run:
python train_model.py
```

This will:
- Train all 4 models (RF, Gradient Boosting, Logistic Regression, SVM)
- Print accuracy, AUC, cross-validation scores, and a full classification report
- Save the best model to `saved_model/model.pkl`
- Print a confusion matrix and feature importances

---

## 🖥️ Running the Web App

**The only way to run the app is to open the HTML file in a browser.**

```bash
# Option 1 — Just double-click the file
heart_disease_predictor.html

# Option 2 — Open from terminal (macOS)
open heart_disease_predictor.html

# Option 3 — Open from terminal (Linux)
xdg-open heart_disease_predictor.html

# Option 4 — Use VS Code Live Server extension
# Right-click the file → "Open with Live Server"
```

> ⚠️ **VS Code's built-in preview won't work** for this file because the embedded model JSON is large. Always open it in a real browser like Chrome, Firefox, or Edge.

---

## 📦 Dependencies

### For the web app
None. It's a single HTML file.

### For retraining
```
Python 3.7+
scikit-learn
pandas
numpy
joblib
```

---

## ⚠️ Disclaimer

This tool is strictly for **educational and research purposes**. It is not a medical device and should not be used to make clinical decisions or replace professional medical advice. Always consult a qualified healthcare provider for any health-related concerns.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙌 Acknowledgements

- Dataset: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/heart+disease) — Cleveland Clinic Foundation
- ML: [scikit-learn](https://scikit-learn.org/)
- Built with ❤️ and a bit of JavaScript wizardry
