# Personality-Based Drug Use Prediction

## Overview
Can personality traits predict drug use? This project uses the UCI Drug Consumption dataset (1,885 individuals) to investigate whether Big Five personality scores and demographic characteristics can classify individuals as users or non-users across three drug categories: stimulants (amphetamines, cocaine, crack), depressants (benzodiazepines, heroin, methadone), and hallucinogens (LSD, mushrooms, ketamine, ecstasy). Two models — Logistic Regression and Random Forest — are compared using proper cross-validation, SMOTE for class balancing, and leakage-free feature pipelines.

## Key Results
- Hallucinogen use was the most predictable category, with Random Forest achieving AUC-ROC of 0.867 and 79% accuracy
- Stimulant and depressant prediction achieved AUC-ROC scores of approximately 0.80, with Logistic Regression slightly outperforming Random Forest for these categories
- Sensation Seeking (SS) and Openness to Experience (Oscore) consistently ranked as the most important predictors across all drug categories
- Earlier model versions revealed data leakage (near-perfect AUC), which was identified and resolved by removing other drug usage labels as predictors

## Methods
- **Feature Engineering**: Grouped 18 individual drugs into 3 categories (stimulants, depressants, hallucinogens), excluding legal substances. Converted 7-class usage levels to binary (user vs. non-user)
- **Modeling**: Logistic Regression and Random Forest classifiers, both wrapped in scikit-learn pipelines with StandardScaler and SMOTE (applied only within training folds to prevent leakage)
- **Evaluation**: 5-fold stratified cross-validation on training data, with held-out test set evaluation. Metrics include ROC AUC, PR AUC, accuracy, classification reports, confusion matrices, ROC curves, PR curves, and feature importance plots
- **Leakage Prevention**: All individual drug usage columns and other target labels are dropped before training to ensure predictions are based solely on personality and demographics

## Dataset
UCI Drug Consumption dataset (ID: 373), fetched via the `ucimlrepo` library. Contains 1,885 records with 12 demographic and personality features and usage levels for 18 substances. No separate data file is needed — the notebook fetches the data directly from UCI.

## Tools
Python, scikit-learn, imblearn (SMOTE), pandas, numpy, matplotlib, seaborn, ucimlrepo

## Repository Contents
```
personality-drug-use-prediction/
├── Nana_Noda-Morgan_DSC630_TermProject.ipynb              # Full analysis notebook
├── Nana_Noda-Morgan_DSC630_TermProject_FinalNotebook.pdf  # Rendered notebook output
├── Nana_Noda-Morgan_DSC630-T302_Project_Final.docx        # Project write-up
├── PredictiveAnalytics_PersonalityPredictDrugUse_Powerpoint.pdf  # Presentation
└── README.md
```

## How to Run
1. Install dependencies: `pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn ucimlrepo`
2. Open the `.ipynb` notebook in Jupyter or VS Code and run all cells. The dataset is fetched automatically from the UCI repository.
