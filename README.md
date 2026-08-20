# Android Games — Hit Game Predictor

A machine learning project that predicts whether an Android game will become a "hit" based on its market, monetization, and engagement metrics. The project includes a full data pipeline — cleaning, feature engineering, feature selection — and trains/compares multiple classification models.

## Project Structure

```
nti-project-main/
├── knkm.ipynb                        # Core notebook: EDA, cleaning, feature engineering, modeling
├── android_games_modeling_v3.ipynb   # Extended modeling notebook
├── app.py                            # Interactive Streamlit app for the same pipeline
└── requirements.txt
```

## Overview

The pipeline takes a raw dataset (`android_games_eda_ready.csv`) and walks through:

1. **Data Cleaning** — drops ID-like columns, removes duplicates, handles missing soft-launch dates, removes leakage-prone columns (post-launch revenue, downloads, etc.).
2. **Train/Test Split** — stratified split on the target `is_hit_game`.
3. **Missing Value Imputation** — median imputation (fit on train only) for numeric fields like marketing spend, CPI, and ARPPU.
4. **Feature Engineering** — developer frequency encoding, categorical frequency encoding.
5. **Outlier Handling** — IQR-based capping on numeric features.
6. **Transformations** — log1p transform on skewed features, standard scaling on the rest.
7. **Feature Selection** — removes multicollinear and weakly-correlated features, ranks remaining features with Mutual Information.
8. **Model Training & Comparison** — trains and compares:
   - Logistic Regression
   - Random Forest
   - XGBoost

Each model is evaluated with Accuracy, Precision, Recall, F1-score, and a confusion matrix, with the best model selected by F1-score.

## Running the App

```bash
pip install -r requirements.txt
streamlit run app.py
```

Upload `android_games_eda_ready.csv` from the sidebar to run the pipeline interactively — every step (cleaning, splitting, feature engineering, model training) is configurable and visualized in the browser.

## Running the Notebooks

Open `knkm.ipynb` or `android_games_modeling_v3.ipynb` in Jupyter Notebook, JupyterLab, or VS Code.

## Requirements

```
pandas
numpy
matplotlib
seaborn
streamlit
scikit-learn
xgboost
```

## Target Variable

`is_hit_game` — binary label indicating whether the game achieved hit status.

## License

For educational/portfolio use.
