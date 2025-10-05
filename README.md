# Credit Risk Predictor

Streamlit application and supporting notebooks for predicting German credit risk using multiple machine-learning models.

## Project Structure
- `app.py` – Streamlit UI that loads the production LightGBM model and encoders.
- `credit_risk_analysis.ipynb` – end-to-end notebook covering data prep, modeling, evaluation, and artifact export.
- `german_credit_data.csv` – source dataset (Kaggle: *German Credit Risk*).
- `LightGBM_credit_model.pkl` – final deployed classifier.
- `*_encoder.pkl` – label encoders for categorical inputs and target.

## Environment Setup
```sh
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Running Locally
```sh
streamlit run app.py
```
Keep the `.pkl` artifacts in the same directory as `app.py`.

## Notebook Walkthrough (`credit_risk_analysis.ipynb`)
1. **Exploratory Analysis**
   - Loaded dataset, inspected schema, summary stats, and class balance of `Risk`.
   - Visualizations: target distribution, age histogram/boxplot, credit amount by risk, categorical feature bars.

2. **Pre-processing**
   - Dropped missing rows and the redundant `Unnamed: 0` column.
  - Label-encoded categorical predictors (`Sex`, `Housing`, `Saving accounts`, `Checking account`) and persisted encoders with `joblib`.
   - Label-encoded `Risk` target and saved `target_encoder.pkl`.
   - Train/test split: 80% train, 20% test, stratified on `Risk`.

3. **Model Training (five algorithms)**
   | Model | Library | Key Hyperparameters | Notes |
   |-------|---------|---------------------|-------|
   | Logistic Regression | `sklearn.linear_model` | `max_iter=1000`, `random_state=42` | Linear baseline for comparison. |
   | Random Forest | `sklearn.ensemble` | `n_estimators=200`, `max_depth=10`, `random_state=42` | Captures non-linearities via bagging. |
   | Gradient Boosting | `sklearn.ensemble` | `n_estimators=200`, `learning_rate=0.1`, `random_state=42` | Boosting alternative with shallow trees. |
   | XGBoost | `xgboost` | `n_estimators=200`, `learning_rate=0.1`, `max_depth=7`, `eval_metric='logloss'` | Gradient boosting with regularization. |
   | LightGBM | `lightgbm` | `n_estimators=200`, `learning_rate=0.1`, `max_depth=7`, `verbose=-1`, `random_state=42` | Fast gradient boosting, ultimately deployed. |

   - Training loop stored accuracy for each model and kept predictions for downstream metrics.

4. **Evaluation**
   - Accuracy comparison table and bar chart across all models (see notebook output cell `comparison_df`).
   - Detailed metrics for the best-performing model (precision, recall, F1-score) plus confusion matrix heatmap.
   - Aggregated metrics table (`metrics_df`) to contrast all algorithms side by side.

5. **Model Selection & Export**
   - Chose the highest-accuracy model (LightGBM) and exported as `LightGBM_credit_model.pkl`.
   - Saved encoder objects to guarantee consistent preprocessing during inference.

## Deployment Notes
1. Commit code, dataset, and model artifacts to the GitHub repository `ML_CIPAT_Project`.
2. On Streamlit Community Cloud:
   - Connect GitHub, select the repo/branch.
   - Set `app.py` as the entry point.
   - Provide required secrets or environment variables if introduced later.

## Reproducing Results
- Re-run `credit_risk_analysis.ipynb` to regenerate metrics and model artifacts.
- Metrics printed in the notebook (accuracy, precision, recall, F1) reference the same train/test split; reruns with a different seed may vary slightly.

## License
Add your preferred license (e.g., MIT) before publishing the repository.

