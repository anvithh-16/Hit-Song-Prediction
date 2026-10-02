import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline


FEATURE_COLUMNS = [
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms",
    "time_signature",
    "chorus_hit",
    "sections",
]

TARGET_COLUMN = "target"


def prepare_data(data):
    """Separate features and target and create a stratified train/test split."""

    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


def create_preprocessor():
    """Create the standardization and PCA preprocessing pipeline."""

    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95)),
    ])


def analyze_pca(preprocessor):
    """
    Analyze PCA variance and feature loadings.

    This function is intended for exploratory analysis and is
    not called during the main project pipeline.
    """

    scaler = preprocessor.named_steps["scaler"]
    pca = preprocessor.named_steps["pca"]

    print("\nPCA Analysis")
    print("-" * 40)

    print("Variance explained by each component:")

    for i, variance in enumerate(
        pca.explained_variance_ratio_,
        start=1
    ):
        print(f"PC{i}: {variance:.4f}")

    print(
        f"\nTotal explained variance: "
        f"{pca.explained_variance_ratio_.sum():.4f}"
    )

    print("\nOriginal feature influence:")

    feature_importance = pd.DataFrame(
        pca.components_.T,
        index=FEATURE_COLUMNS,
        columns=[
            f"PC{i}"
            for i in range(1, pca.n_components_ + 1)
        ]
    )

    print(feature_importance.round(3))