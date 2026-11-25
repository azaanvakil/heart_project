# Heart Disease Prediction

An end-to-end binary classification project using the UCI Heart Disease dataset. It compares Logistic Regression, Random Forest, and XGBoost, tunes the strongest baseline, and saves a preprocessing-plus-model pipeline for the Streamlit app. This educational example is not a clinical decision tool.

## Dataset

- [Kaggle mirror](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)
- [Original UCI Heart Disease dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)

The target is `num > 0`; `id` and `dataset` are excluded.

## Run the notebook

Use Python 3.11. Install dependencies, then execute all cells from the project directory:

```bash
python -m pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

The final notebook cell creates `model.pkl`.

## Run the app

```bash
streamlit run app.py
```

Enter values for the 13 model features to view a predicted probability and a Low/High risk label. The app's disclaimer is: “Educational project, not medical advice.”

## Results

Run the notebook to populate this table from the stratified test split (random state 42). Metrics are rounded to three decimals.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.842 | 0.841 | 0.882 | 0.861 | 0.904 |
| Random Forest | 0.864 | 0.860 | 0.902 | 0.880 | 0.924 |
| XGBoost | 0.859 | 0.858 | 0.892 | 0.875 | 0.910 |
| Tuned Random Forest | 0.870 | 0.868 | 0.902 | 0.885 | 0.925 |

## Screenshots

_Add notebook plots and a Streamlit app screenshot here._
