# 💳 Paytm Fraud Detection: Machine Learning Project

A machine learning project that detects **fraudulent digital payment transactions**. It compares four classifiers (Decision Tree, Random Forest, AdaBoost, and XGBoost) on a dataset of 70,000 transactions and selects **XGBoost** as the best model.

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-green)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)

---

## 📌 Problem Statement

Digital payment platforms process millions of transactions every day, and a small share of them are fraudulent. Missing a fraudulent transaction costs money and customer trust, while wrongly blocking a genuine one frustrates users.

**Goal:** build a classification model that flags fraudulent transactions with **high recall** (catch as much fraud as possible) while keeping precision reasonable.

---

## 📂 Dataset

- **Size:** 70,000 transactions × 27 columns
- **Target:** `isFraud` (1 = fraud, 0 = genuine)
- **Source:** *(add the source here, e.g. Kaggle link or "synthetic dataset provided during the course")*

| Group | Features |
|---|---|
| Transaction | amount, transaction time/day/month/hour, weekend flag, merchant category, payment method |
| Account | sender and receiver account age, account balance, beneficiary count, KYC verified |
| Behaviour | transaction frequency (30 days), average transaction amount (30 days), failed attempts, login time gap |
| Device & location | device type, device ID, IP address, city, state, VPN used, new-device login |

---

## 🔄 Project Workflow

1. **Data understanding:** shape, data types, distributions, and correlation heatmap
2. **Missing value handling:** group-wise median imputation (for example by payment method, device, and transaction frequency) and "Unknown" labels for missing categories
3. **Outlier analysis:** box plots on amount, account balance, and login time gap
4. **Feature engineering:** count features such as `ip_txn_count` and `device_txn_count`, plus a transaction-frequency group (`txn_freq_group`)
5. **Encoding:** label encoding of categorical columns; dropped identifiers (`transaction_id`, date, time, device ID, IP address)
6. **Train/test split:** 80/20 stratified split (56,000 train / 14,000 test, `random_state=42`)
7. **Model training:** Decision Tree, Random Forest, AdaBoost, and XGBoost
8. **Hyperparameter tuning:** `GridSearchCV` with 5-fold cross-validation (72 parameter combinations, 360 fits) for XGBoost
9. **Evaluation:** accuracy, precision, recall, F1-score, confusion matrix, ROC curve, and ROC AUC

---

## 📈 Model Performance

All results are on the held-out test set of **14,000 transactions** (10,375 genuine and 3,625 fraud).

| Model | Accuracy | Fraud Precision | Fraud Recall | Fraud F1 | ROC AUC |
|---|---|---|---|---|---|
| Decision Tree | 82.61% | n/a | n/a | n/a | n/a |
| AdaBoost | 87.68% | n/a | n/a | n/a | n/a |
| Random Forest (100 trees) | 90.03% | 0.76 | 0.90 | 0.82 | 0.8998 |
| **XGBoost** ⭐ | **90.00%** | **0.76** | **0.90** | **0.82** | **0.9044** |

Random Forest out-of-bag (OOB) score: **0.8996**, consistent with its test accuracy, which indicates the model is not overfitting.

### Classification report (XGBoost)

```
              precision    recall  f1-score   support

           0       0.96      0.90      0.93     10375
           1       0.76      0.90      0.82      3625

    accuracy                           0.90     14000
   macro avg       0.86      0.90      0.88     14000
weighted avg       0.91      0.90      0.90     14000
```

### 🏆 Best Model: XGBoost

**XGBoost was selected as the final model because:**

- It has the **highest ROC AUC (0.9044)**, so it separates fraud from genuine transactions better across all decision thresholds than Random Forest (0.8998).
- Its accuracy (90.0%) and fraud-class recall (0.90) match the best of the other models.
- Boosted trees handle the mix of numeric and encoded categorical features well and are fast to train and score.
- A `GridSearchCV` over 72 parameter combinations (5-fold CV) returned the same parameters as the baseline, which shows the model was already well configured:

```python
{'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1,
 'subsample': 0.8, 'colsample_bytree': 1.0}
```

> **Note:** the gap between XGBoost and Random Forest is small (about 0.005 ROC AUC), so both are strong. XGBoost is chosen for its slightly better ranking quality (AUC) and lighter, simpler tuning.

---

## 🔍 Key Insights

Top predictors of fraud (Random Forest feature importance):

| Rank | Feature | Importance |
|---|---|---|
| 1 | Transaction frequency (30 days) | 31.0% |
| 2 | Transaction frequency group | 18.4% |
| 3 | Login time gap (hours) | 8.7% |
| 4 | Transaction amount | 5.5% |
| 5 | Failed attempts | 3.8% |

- **Transaction behaviour matters most:** unusual transaction frequency is the strongest signal of fraud.
- **Login patterns and amounts** add further signal, along with failed attempts and account balance.
- The model **catches about 90% of fraud cases** (recall 0.90), at the cost of some false alarms (fraud precision 0.76).

---

## 🛠️ Tech Stack

- **Language:** Python
- **Data:** pandas, NumPy
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Machine learning:** scikit-learn, XGBoost
- **Environment:** Jupyter Notebook

---

## 📁 Repository Structure

```
paytm-fraud-detection/
├── notebooks/
│   └── paytm_fraud_detection.ipynb
├── data/
│   └── paytm_fraud_detection_dataset.csv
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run

```bash
# 1. Clone the repository
git clone https://github.com/payalpatole-2004/paytm-fraud-detection.git
cd paytm-fraud-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Open the notebook
jupyter notebook notebooks/paytm_fraud_detection.ipynb
```

Make sure the dataset path in the notebook matches where the CSV is stored (for example `../data/paytm_fraud_detection_dataset.csv`).

---

## 🚀 Future Improvements

- Handle class imbalance with SMOTE or class weights and compare precision/recall trade-offs
- Tune the decision threshold to balance fraud recall against false alarms
- Add SHAP explanations to show why each transaction is flagged
- Build a Streamlit app for real-time fraud prediction
- Save the trained model (`joblib`) and serve it through a REST API (Flask)

---

## 👩‍💻 Author

**Payal Patole**
Aspiring Data Analyst / Data Scientist | Pune, India
🔗 GitHub: [payalpatole-2004](https://github.com/payalpatole-2004)
