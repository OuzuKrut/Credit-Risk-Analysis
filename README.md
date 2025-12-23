# Credit-Risk-Analysis

# 🏦 Cost-Sensitive Credit Risk Analysis

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Library](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Status-Academic_Project-green)

## 📌 Project Overview
This project implements a machine learning solution to assess **Credit Risk** using the famous **UCI Statlog (German Credit Data)** dataset.

Unlike standard classification tasks, this project focuses on a **Cost-Sensitive Learning** approach. In the domain of credit risk, classifying a "Bad" customer as "Good" (False Negative) is financially far more damaging than the reverse. Therefore, this model is optimized to minimize the total financial cost based on a **1:5 Cost Matrix**.

## 🎯 Business Problem & Objective
* **Dataset:** 1000 entries with 20 categorical/symbolic attributes (Checking Status, Credit History, Purpose, etc.).
* **Challenge:** The dataset is imbalanced (70% Good / 30% Bad), and standard accuracy metrics are misleading.
* **Cost Matrix:** * Misclassifying a **Good** customer as Bad = **1 Unit Cost** (Opportunity Loss)
    * Misclassifying a **Bad** customer as Good = **5 Unit Cost** (Direct Financial Loss)
* **Goal:** To build a robust decision support system that prioritizes **Recall (Sensitivity)** for the "Bad Credit" class, ensuring high-risk applicants are identified.

## ⚙️ Methodology

### 1. Data Preprocessing & Integrity
* **Mapping:** Converted cryptic categorical codes (e.g., `A11`, `A34`) into meaningful business labels based on the dataset documentation.
* **Leakage Prevention:** Applied `StandardScaler` only on the training set and transformed the test set using learned parameters to strictly prevent data leakage.
* **Stratified Splitting:** Maintained the class distribution ratio during train-test splits.

### 2. Feature Engineering & Selection
* **RFE (Recursive Feature Elimination):** Utilized RFE with Logistic Regression to select the top **15 most significant features** impacting credit risk.
* **Key Drivers Identified:** Checking Account Status, Credit Duration, and Credit History proved to be the strongest predictors.

### 3. Cost-Sensitive Modeling
* **Algorithm:** Logistic Regression (`solver='liblinear'`).
* **Class Weights:** Implemented a `{0:1, 1:5}` class weight ratio to penalize misclassification of bad credits heavily.
* **Threshold Tuning:** Instead of the default 0.50 probability threshold, I implemented a loop to find the **Optimal Decision Threshold** that minimizes the specific cost function `(FP * 1) + (FN * 5)`.

## 📊 Key Results

The model was evaluated based on its ability to minimize financial cost and identify high-risk customers.

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Optimal Threshold** | 0.50 | Probability cutoff. Lower than 0.5 to act conservatively against risk. |
| **Recall (Bad Credit)** | 0.93 | The model captures this percentage of all actual defaulters. |
| **ROC-AUC Score** | 0.7373 | Indicates the model's ability to distinguish between classes. |
| **Gini Coefficient** | 0.4746 | Banking industry standard metric `(2*AUC - 1)`. |
| **KS Statistic** | 0.4452 | Measures the separation distance between Good/Bad distributions. |

### Visualization
The chart below shows the top 15 features affecting credit risk. **Positive (Red) bars** indicate factors that increase risk, while **Negative (Green) bars** indicate factors that increase reliability.

*(Please ensure `german_credit_analysis.png` is in the repository)*
![Feature Importance](german_credit_analysis.png)

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/german-credit-risk.git](https://github.com/YOUR_USERNAME/german-credit-risk.git)
2. Install dependencies:
   pip install pandas numpy scikit-learn matplotlib scipy
3. Run the analysis:
   python analysis.py

📂 Project Structure
german-credit-risk/
├── analysis.py               # Main Python script
├── german_credit_analysis.png # Feature Importance Chart
├── requirements.txt          # Dependencies
└── README.md                 # Project Documentation

👨‍💻 Author
Oğuz Kurt Statistics Student @ Yıldız Technical University

www.linkedin.com/in/oğuzkurt
