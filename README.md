# NTI Project — Android Games Hit Predictor 🎮

A machine learning project that predicts whether an Android game will become a "hit," based on its data (downloads, revenue, ratings, genre, monetization model, etc.).

## 📂 Project Structure

```
nti-project/
├── README.md
├── requirements.txt
├── android_games_eda_ready.csv      # Dataset
├── knkm.ipynb                       # Main notebook: EDA + model training
└── app.py                           # Interactive Streamlit app with the same pipeline
```

## 🗂️ Dataset

`android_games_eda_ready.csv` is sourced from Kaggle:
[Android Games EDA Ready Dataset](https://www.kaggle.com/datasets/tsmgofficial/android-games-eda-ready-dataset)

## 🔄 Pipeline Steps

1. **Data cleaning**: drop ID-like columns and duplicates, drop data-leakage columns (e.g., actual revenue and downloads).
2. **Train/test split** with stratification to preserve class balance.
3. **Missing value imputation** using the median.
4. **Feature engineering**: convert developer name into a developer-game-count feature, frequency-encode categorical columns.
5. **Outlier handling** using the IQR method.
6. **Feature selection**: drop columns with high multicollinearity or weak correlation with the target, and display Mutual Information scores.
7. **Model training and comparison**: Logistic Regression, Random Forest, and XGBoost — reporting Accuracy/Precision/Recall/F1 for each and selecting the best one.

## 🚀 Running the Project

```bash
pip install -r requirements.txt
streamlit run app.py
```

If the `streamlit` command is not recognized (common on Windows), run it as a Python module instead:

```bash
python -m streamlit run app.py
```

After launching, upload `android_games_eda_ready.csv` from the sidebar in the app.

## 📓 Notebooks

- `knkm.ipynb`: original full analysis and experiments.
- `android_games_modeling_v3.ipynb`: additional analysis/modeling notebook.

## 🤖 Models

| Model | Notes |
|---|---|
| Logistic Regression | `class_weight='balanced'` |
| Random Forest | `n_estimators=200`, `max_depth=20` |
| XGBoost | `scale_pos_weight` to balance classes |

The best model is automatically selected based on the highest F1-score.
