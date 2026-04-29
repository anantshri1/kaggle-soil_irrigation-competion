# Kaggle Competition — Soil Irrigation Need Classification

Multiclass classification task predicting agricultural irrigation need (**Low / Medium / High**) from soil and crop features, submitted to the [Kaggle Playground Series S6E4](https://www.kaggle.com/competitions/playground-series-s6e4/leaderboard) competition. The dataset was synthetically generated from a deep learning model trained on the [Irrigation Water Requirement Prediction dataset](https://www.kaggle.com/datasets/miadul/irrigation-water-requirement-prediction-dataset/data).

## The Challenge

The target variable is severely imbalanced — **High** irrigation need represents only ~3% of training samples, with **Low** at ~67% and **Medium** at ~30%. A naive model predicting the majority class would achieve deceptively high accuracy while being practically useless for identifying high-need cases. The core challenge is building a model that handles this imbalance effectively.

## Approaches

Four approaches are explored and compared:

| Model | Validation Accuracy |
|---|---|
| Undersampling + Random Forest | ~0.970 |
| Balanced Random Forest (no SMOTE) | 0.9849 |
| Balanced Random Forest + SMOTE | 0.9845 |
| XGBoost | 0.9844 |

> Note: The undersampling model is trained on a heavily reduced dataset (matched to the minority class size) and evaluated on a different split, so its accuracy is not directly comparable to the other three models.

**Kaggle leaderboard score: 95.9%** (XGBoost submission) **95.8%** (Random Forest without SMOTE submission)

## Key Techniques

- `ColumnTransformer` preprocessing pipelines with `OneHotEncoder` and `StandardScaler` > Note: It may have been easier to evaluate this with `LabelEncoder` rather than `OneHotEncoder` due to the large number of labels.
- Majority class undersampling as a naive baseline
- `class_weight='balanced'` in Random Forest to penalise minority class misclassification
- SMOTE oversampling via an `imblearn` pipeline
- XGBoost multiclass classification (`multi:softprob`)
- Stratified train/test splits to preserve class distribution

## How to Run

1. Download `train_soil.csv` and `test_soil.csv` from the [competition page](https://www.kaggle.com/competitions/playground-series-s6e4/leaderboard) and place them in the root directory.

2. Install dependencies:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost
```

3. Run `Kaggle_Soil-Irrigation_Competition.ipynb` top to bottom.

## Tech Stack

Python · pandas · scikit-learn · imbalanced-learn · XGBoost · matplotlib · seaborn
