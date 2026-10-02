from pathlib import Path

import pandas as pd


RAW_DATA_DIR = Path("data/raw")

REQUIRED_COLUMNS = {
    "track",
    "artist",
    "uri",
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
    "target",
}


def load_dataset():
    """Load and combine all decade-wise CSV files."""

    csv_files = sorted(
        RAW_DATA_DIR.glob("dataset-of-*.csv")
    )

    if not csv_files:
        raise FileNotFoundError(
            f"No dataset files found in {RAW_DATA_DIR}"
        )

    dataframes = []

    for file in csv_files:
        df = pd.read_csv(file)
        dataframes.append(df)

    return pd.concat(
        dataframes,
        ignore_index=True
    )


def validate_dataset(data):
    """Validate the structure and target labels of the dataset."""

    missing_columns = REQUIRED_COLUMNS - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if data["target"].isna().any():
        raise ValueError(
            "Target column contains missing values."
        )

    if not data["target"].isin([0, 1]).all():
        raise ValueError(
            "Target column must contain only 0 and 1."
        )

    return True