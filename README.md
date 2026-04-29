# kaggle-soil_irrigation-competion
Multiclass classification task predicting agricultural irrigation need (Low / Medium / High) from soil and crop features. Dataset sourced from a Kaggle competition based on a synthetically generated version of the Irrigation Water Requirement Prediction dataset.

The core challenge is class imbalance — High irrigation need represents only ~3% of training samples. Three approaches are explored and compared: majority class undersampling, balanced Random Forest without resampling, and SMOTE oversampling via an imblearn pipeline. XGBoost is then applied as a final model, achieving 95.919% accuracy on the Kaggle leaderboard.
Key techniques: ColumnTransformer preprocessing pipelines, SMOTE via imblearn, balanced class weighting, XGBoost multiclass classification, stratified train/test split.
