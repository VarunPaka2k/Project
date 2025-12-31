---

# 📉 Telco Customer Churn Prediction

Predict customer churn in the telecommunications industry using machine learning, class imbalance handling, and ensemble models.

---

## 📌 Overview

Customer churn is a major challenge for telecom companies, as retaining customers is more cost-effective than acquiring new ones.
This project predicts whether a customer is likely to churn by analyzing demographic information, service usage, contract details, and billing behavior.

The project follows a **two-notebook pipeline**:

* Exploratory Data Analysis (EDA)
* End-to-end modeling, evaluation, SMOTE, and ensemble learning

---

## 📁 Project Structure

```
telco-customer-churn/
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   ├── Clean_Telco-Customer-Churn_data.csv
│   ├── processed_Telco-Customer-Churn_data.csv
│   └── TelcoChurn_Top25_XGB.csv
├── notebooks/
│   ├── 01_EDA.ipynb
│   └── churn_modeling_and_evaluation.ipynb
├── models/
│   ├── LogisticRegression_BestModel.sav
│   ├── RandomForest_BestModel.sav
│   ├── XGBoost_BestModel.sav
│   └── BEST_Stacking_SMOTE.sav
├── images/
│   ├── churn_distribution.jpg
│   ├── top3_models_heatmap.jpg
│   └── Heatmap_Comparison_SMOTE.jpg
├── requirements.txt
├── README.md
```

---

## ⚙️ Features

* Exploratory Data Analysis with visual insights
* Feature engineering and preprocessing
* Supervised models: Logistic Regression, Random Forest, XGBoost
* Hyperparameter tuning using GridSearchCV
* Automatic probability threshold optimization
* Class imbalance handling using **SMOTE-Tomek**
* Ensemble learning with **Stacking**
* Model comparison using tables and heatmaps

---

## 📦 Dataset

* **Source:** Kaggle – Telco Customer Churn
* **Link:** [https://www.kaggle.com/datasets/blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
* **Records:** ~7,000 customers
* **Target Variable:** `Churn` (0 = No, 1 = Yes)

---

## 🚀 How to Run

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Run notebooks in order

```
1. 01_EDA.ipynb  
2. churn_modeling_and_evaluation.ipynb
```

---

## 🧠 Modeling Approach

* Data cleaning and preprocessing
* Feature engineering
* Train–test split with stratification
* Model training and tuning
* Threshold optimization (F1-score based)
* SMOTE-Tomek for class imbalance
* Ensemble modeling using stacking
* Model selection based on recall-focused evaluation

---

## 📊 Results Summary

| Model                 | Type   | Recall    | ROC-AUC   | F1        |
| --------------------- | ------ | --------- | --------- | --------- |
| BEST Stacking (SMOTE) | SMOTE  | **0.775** | 0.825     | 0.614     |
| XGBoost               | Normal | 0.773     | **0.842** | **0.637** |
| Random Forest (SMOTE) | SMOTE  | 0.759     | 0.832     | 0.623     |
| Stacking Ensemble     | Normal | 0.671     | 0.842     | 0.633     |

✔ **Final model selected:** BEST Stacking (SMOTE)
✔ **Reason:** Highest recall, suitable for churn-risk detection

---

## 📌 Conclusion

The results show that **handling class imbalance and using ensemble learning significantly improves churn prediction**.
SMOTE-based models improve recall, while stacking ensembles provide balanced performance across metrics.
The final model prioritizes identifying at-risk customers, supporting effective customer retention strategies.

---

## 👨‍💻 Author

**Varun Paka**

---

