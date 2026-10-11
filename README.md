# Hit Song Prediction Using Audio Features

## Overview

This project implements a machine learning framework for predicting whether a
song is a hit or a flop using Spotify audio features.

The project is an adaptation of the methodology presented in the research paper:

**"Predicting Hit Songs Using Repeated Chorus"**

The original paper investigates whether audio characteristics from a repeated
chorus can be used to predict song popularity. Due to the time constraints of
this mini-project, the original data-collection and chorus-extraction pipeline
was not reproduced. Instead, an existing public Spotify audio-feature dataset
was used to implement and evaluate the core machine learning methodology.

---

## Problem Statement

The objective is to build a supervised binary classification system that
predicts whether a song is a hit based on its available audio and
audio-analysis features.

The target variable is:

- `1` — Hit
- `0` — Flop

The dataset defines a hit as a song that appeared on the Billboard Hot-100
weekly list at least once during its respective decade.

---

## Dataset

The project uses the **Spotify Audio Features Hit Predictor Dataset
(1960–2019)**.

The dataset contains songs from six decades:

- 1960s
- 1970s
- 1980s
- 1990s
- 2000s
- 2010s

After combining the decade-wise CSV files, the dataset contains:

- **41,106 songs**
- **19 columns**
- **15 input features**
- **1 binary target variable**

The dataset is balanced, containing 20,553 hit songs and 20,553 flop songs.

The raw dataset is not included in the Git repository because it is excluded
through `.gitignore`.

---

## Features

The following 15 features are used for prediction:

- Danceability
- Energy
- Key
- Loudness
- Mode
- Speechiness
- Acousticness
- Instrumentalness
- Liveness
- Valence
- Tempo
- Duration
- Time Signature
- Chorus Hit
- Sections

The following identifying fields are excluded from model training:

- Track
- Artist
- URI

This ensures that the models use audio/audio-analysis characteristics rather
than artist or track identity.

---

## Methodology

The implemented pipeline consists of the following stages:

1. Load and combine the six decade-wise datasets.
2. Validate the dataset structure and target labels.
3. Separate input features and target variable.
4. Create an 80/20 stratified train-test split.
5. Standardize the numerical features.
6. Apply Principal Component Analysis (PCA).
7. Train multiple machine learning classification models.
8. Evaluate the models using 5-fold stratified cross-validation.
9. Train the models on the complete training set.
10. Evaluate final performance on the held-out test set.
11. Generate confusion matrices, ROC curves, and model comparison plots.

### PCA

The 15 input features are standardized before applying PCA.

PCA retains enough components to explain at least 95% of the variance.
For this dataset:

- Original features: **15**
- Principal components retained: **12**
- Variance explained: **96.0%**

The PCA analysis is also used to examine how the original audio features
contribute to the principal components.

The PCA loading analysis is exploratory and should not be interpreted as
feature importance for predicting hit songs.

---

## Machine Learning Models

The following models are implemented:

1. Logistic Regression
2. Linear Discriminant Analysis with shrinkage
3. Linear Support Vector Machine
4. RBF Support Vector Machine
5. Polynomial Support Vector Machine
6. Random Forest
7. Gradient Boosting
8. Neural Network

All models use the same preprocessing pipeline so that their results can be
compared under consistent preprocessing conditions.

---
## Running the Project

### 1. Prerequisites

- Python 3.10 or later recommended.
- Git installed on your system.

### 2. Clone the Repository

```bash
git clone https://github.com/anvithh-16/hit-song-prediction.git
cd hit-song-prediction
```

### 3. Set Up a Virtual Environment

Create and activate a virtual environment:

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows**
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Download and Prepare the Dataset

The dataset files are excluded from the Git repository. Obtain the Spotify Audio Features Hit Predictor Dataset (1960–2019) and place the six decade-wise CSV files inside the `data/raw/` directory:

```text
data/raw/
├── dataset-of-00s.csv
├── dataset-of-10s.csv
├── dataset-of-60s.csv
├── dataset-of-70s.csv
├── dataset-of-80s.csv
└── dataset-of-90s.csv
```

The CSV filenames must match the `dataset-of-*.csv` pattern.

### 6. Run the Project

From the repository root, execute:

```bash
python main.py
```

The script loads and validates the dataset, prepares the features, performs 5-fold stratified cross-validation, trains and evaluates eight machine learning models, and generates performance visualizations.

### 7. View the Results

After execution, the generated results are available in:

- `results/tables/cross_validation_results.csv` — cross-validation metrics.
- `results/tables/model_comparison.csv` — held-out test-set metrics.
- `results/figures/confusion_matrices.png` — confusion matrices for all models.
- `results/figures/roc_curves.png` — ROC curves and AUC scores.
- `results/figures/model_comparison.png` — model performance comparison.

**Note:** Run `python main.py` from the repository root so that the relative dataset and output paths resolve correctly.

---

## Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC-AUC

A **5-fold stratified cross-validation** procedure is performed on the training
set to assess performance stability across different splits.

The final reported test metrics are calculated on a separate held-out test
set that is not used during model training.

---

## Results

### 5-Fold Cross-Validation

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.7228 | 0.6887 | 0.8131 | 0.7457 |
| LDA | 0.7173 | 0.6731 | 0.8449 | 0.7493 |
| Linear SVM | 0.7193 | 0.6759 | 0.8426 | 0.7501 |
| RBF SVM | 0.7680 | 0.7222 | 0.8712 | 0.7897 |
| Polynomial SVM | 0.7349 | 0.6744 | 0.9083 | 0.7741 |
| Random Forest | 0.7615 | 0.7361 | 0.8155 | 0.7737 |
| Gradient Boosting | 0.7499 | 0.7141 | 0.8340 | 0.7693 |
| Neural Network | 0.7709 | 0.7391 | 0.8385 | 0.7853 |

### Held-Out Test Set

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.7302 | 0.6930 | 0.8268 | 0.7540 |
| LDA | 0.7235 | 0.6765 | 0.8567 | 0.7560 |
| Linear SVM | 0.7221 | 0.6746 | 0.8579 | 0.7553 |
| RBF SVM | 0.7733 | 0.7277 | 0.8735 | 0.7939 |
| Polynomial SVM | 0.7378 | 0.6770 | 0.9095 | 0.7762 |
| Random Forest | 0.7693 | 0.7438 | 0.8215 | 0.7807 |
| Gradient Boosting | 0.7576 | 0.7204 | 0.8419 | 0.7764 |
| Neural Network | 0.7728 | 0.7498 | 0.8188 | 0.7828 |

The ROC-AUC values on the held-out test set were:

| Model | ROC-AUC |
|---|---:|
| Logistic Regression | 0.800 |
| LDA | 0.798 |
| Linear SVM | 0.800 |
| RBF SVM | 0.844 |
| Polynomial SVM | 0.807 |
| Random Forest | 0.841 |
| Gradient Boosting | 0.828 |
| Neural Network | 0.847 |

The cross-validation and held-out test results are reasonably consistent,
indicating similar performance across the evaluated data splits.

---

## Results Visualization

The project automatically generates:

- Confusion matrices for all models
- ROC curves for all models
- Model performance comparison

These are saved in:

```text
results/figures/
```

Numerical Results are saved in:

```text
results/tables/
