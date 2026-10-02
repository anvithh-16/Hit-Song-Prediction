from src.data_loader import load_dataset, validate_dataset
from src.preprocessing import prepare_data
from src.models import build_models
from src.evaluation import (
    cross_validate_models,
    train_and_evaluate,
    plot_confusion_matrices,
    plot_roc_curves,
    plot_model_comparison,
)


def main():
    print("=" * 70)
    print("HIT SONG PREDICTION USING AUDIO FEATURES")
    print("=" * 70)

    # --------------------------------------------------
    # 1. Load and validate dataset
    # --------------------------------------------------
    print("\n[1/5] Loading dataset...")

    data = load_dataset()
    validate_dataset(data)

    print(f"Dataset shape: {data.shape}")

    # --------------------------------------------------
    # 2. Prepare data
    # --------------------------------------------------
    print("\n[2/5] Preparing data...")

    X_train, X_test, y_train, y_test = prepare_data(data)

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Number of features: {X_train.shape[1]}")

    # --------------------------------------------------
    # 3. Build models
    # --------------------------------------------------
    print("\n[3/5] Building models...")

    models = build_models()

    print(f"Models: {len(models)}")

    # --------------------------------------------------
    # 4. Cross-validation
    # --------------------------------------------------
    print("\n[4/5] Running 5-fold cross-validation...")

    cv_results = cross_validate_models(
        models,
        X_train,
        y_train
    )

    print("\n5-FOLD CROSS-VALIDATION RESULTS")
    print("-" * 70)
    print(cv_results.to_string(index=False))

    cv_results.to_csv(
        "results/tables/cross_validation_results.csv",
        index=False
    )

    # --------------------------------------------------
    # 5. Final training and test evaluation
    # --------------------------------------------------
    print("\n[5/5] Training models and evaluating on test set...")

    results = train_and_evaluate(
        models,
        X_train,
        X_test,
        y_train,
        y_test
    )

    print("\nMODEL COMPARISON")
    print("-" * 70)
    print(results.to_string(index=False))

    results.to_csv(
        "results/tables/model_comparison.csv",
        index=False
    )

    # --------------------------------------------------
    # Generate visualizations
    # --------------------------------------------------
    print("\nGenerating visualizations...")

    plot_confusion_matrices(
        models,
        X_test,
        y_test
    )

    plot_roc_curves(
        models,
        X_test,
        y_test
    )

    plot_model_comparison(
        results
    )

    print("\n" + "=" * 70)
    print("PROJECT COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print("\nGenerated files:")
    print("  results/tables/cross_validation_results.csv")
    print("  results/tables/model_comparison.csv")
    print("  results/figures/confusion_matrices.png")
    print("  results/figures/roc_curves.png")
    print("  results/figures/model_comparison.png")


if __name__ == "__main__":
    main()