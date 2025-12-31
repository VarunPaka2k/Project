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
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv        # Original Kaggle dataset
│   ├── Clean_Telco-Customer-Churn_data.csv         # Cleaned dataset after EDA
│   ├── processed_Telco-Customer-Churn_data.csv     # Fully processed & encoded data
│   └── TelcoChurn_Top25_XGB.csv                    # Top 25 features selected via XGBoost
│
├── notebooks/
│   ├── EDA.ipynb                                  # Exploratory Data Analysis
│   └── churn_modeling_and_evaluation.ipynb        # Modeling, SMOTE, ensembles, evaluation
│
├── models/
│   ├── LogisticRegression_BestModel.sav            # Best Logistic Regression (normal)
│   ├── RandomForest_BestModel.sav                  # Best Random Forest (normal)
│   ├── XGBoost_BestModel.sav                       # Best XGBoost (normal)
│   ├── StackingEnsemble_BestModel.sav              # Stacking ensemble (normal)
│   ├── LogisticRegression_SMOTE.sav                # Logistic Regression trained with SMOTE
│   ├── RandomForest_SMOTE.sav                      # Random Forest trained with SMOTE
│   ├── XGBoost_SMOTE.sav                           # XGBoost trained with SMOTE
│   └── BEST_Stacking_SMOTE.sav                     # Final selected stacking model (SMOTE)
│
├── images/
│   ├── churn_distribution.jpg                     # Churn class distribution
│   ├── Smote.jpg                                  # Class distribution after SMOTE
│   ├── Xg.jpg                                     # XGBoost confusion matrix
│   ├── Smote stacking.jpg                         # Confusion matrix for BEST stacking (SMOTE)
│   ├── Heatmap Comparison - All SMOTE Models.jpg  # Heatmap of all SMOTE model performance
│   └── top3_models_heatmap.jpg                    # Heatmap highlighting top 3 models
│
├── requirements.txt                               # Python dependencies
└── README.md                                      # Project documentation
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

- **Name:** Telco Customer Churn  
- **Primary Source:** IBM Analytics Community  
  https://community.ibm.com/community/user/businessanalytics/blogs/steven-macko/2019/07/11/telco-customer-churn-1113
- **Accessed via:** Kaggle  
  https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- **Records:** ~7,000 customers  
- **Target Variable:** Churn (0 = No, 1 = Yes)

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

