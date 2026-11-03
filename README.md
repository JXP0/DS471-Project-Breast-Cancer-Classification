# DS-471 Machine Learning Project: Breast Cancer Classification

Predicting whether a breast mass is **malignant** or **benign** from 30 cell-nucleus measurements taken from digitized fine-needle aspirate (FNA) images.

**Dataset:** [UCI Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic). It has 569 samples (357 benign, 212 malignant). The notebook loads it with `sklearn.datasets.load_breast_cancer`, which is the same UCI data bundled with scikit-learn, so you don't need to download anything.

## Repository contents

| File | Description |
| --- | --- |
| `DS471_Breast_Cancer_Classification.ipynb` | Full workflow with outputs: cleaning, EDA, feature engineering, splitting, tuning, training/validation curves, evaluation |
| `figures/` | All figures used in the report |
| `results_test.csv` | Final test-set metrics for every model |
| `requirements.txt` | Python dependencies |

## Workflow

1. **Data preparation and EDA.** Checks for missing values, duplicates, zeros and outliers. Looks at the target distribution, feature skew and correlations, and ranks features by mutual information.
2. **Preprocessing** (fitted on the training data only, inside a `Pipeline`): `log1p` on skewed features, a correlation filter (|r| > 0.95) and standard scaling. The Random Forest uses the raw features.
3. **Split:** 70 / 15 / 15 stratified (398 / 85 / 86 rows).
4. **Models:** Logistic Regression, SVM (RBF), Random Forest and a Neural Network (MLP), tuned with 5-fold stratified grid search on F1.
5. **NN training/validation curves:** epoch-by-epoch loss and accuracy, early stopping with the best weights restored, and an unregularised network for comparison.
6. **Test evaluation:** accuracy, precision, recall, F1, ROC-AUC, confusion matrices and ROC curves.

## Results (test set)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.965 | 0.968 | 0.938 | 0.952 | 0.999 |
| SVM (RBF) | 0.977 | 1.000 | 0.938 | 0.968 | 1.000 |
| Random Forest | 0.988 | 1.000 | 0.969 | 0.984 | 0.999 |
| Neural Network (MLP) | 0.977 | 1.000 | 0.938 | 0.968 | 0.999 |

The recommended model is the **SVM (RBF)**. It had the best cross-validation and validation F1 and a perfect test AUC. The Random Forest beat it on the test set by one patient. The report explains the choice in full.

## How to run

```bash
pip install -r requirements.txt
jupyter notebook DS471_Breast_Cancer_Classification.ipynb
```

Or open the notebook in Google Colab and select *Runtime → Run all*. All randomness uses seed 42.

## Team

- Member 1: ID
- Member 2: ID
- Member 3: ID
- Member 4: ID
