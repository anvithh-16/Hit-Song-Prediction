import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
)


def evaluate_model(model, X_test, y_test):
    """Evaluate a trained model on the test set."""

    predictions = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
    }


def train_and_evaluate(models, X_train, X_test, y_train, y_test):
    """Train all models and evaluate them on the test set."""

    results = []

    for name, model in models.items():

        print(f"\nTraining {name}...")

        model.fit(X_train, y_train)

        metrics = evaluate_model(
            model,
            X_test,
            y_test
        )

        results.append({
            "Model": name,
            "Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1 Score": metrics["f1"],
        })

        print(
            f"Accuracy:  {metrics['accuracy']:.4f}\n"
            f"Precision: {metrics['precision']:.4f}\n"
            f"Recall:    {metrics['recall']:.4f}\n"
            f"F1 Score:  {metrics['f1']:.4f}"
        )

    return pd.DataFrame(results)


def cross_validate_models(models, X_train, y_train):
    """Perform 5-fold stratified cross-validation."""

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
    }

    results = []

    for name, model in models.items():

        print(f"\nCross-validating {name}...")

        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )

        results.append({
            "Model": name,

            "CV Accuracy Mean": scores["test_accuracy"].mean(),
            "CV Accuracy Std": scores["test_accuracy"].std(),

            "CV Precision Mean": scores["test_precision"].mean(),
            "CV Precision Std": scores["test_precision"].std(),

            "CV Recall Mean": scores["test_recall"].mean(),
            "CV Recall Std": scores["test_recall"].std(),

            "CV F1 Mean": scores["test_f1"].mean(),
            "CV F1 Std": scores["test_f1"].std(),
        })

    return pd.DataFrame(results)


def plot_confusion_matrices(models, X_test, y_test):
    """Generate confusion matrices for all models."""

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()

    for i, (name, model) in enumerate(models.items()):

        predictions = model.predict(X_test)
        cm = confusion_matrix(y_test, predictions)

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=axes[i]
        )

        axes[i].set_title(name)
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")

    plt.tight_layout()

    plt.savefig(
        "results/figures/confusion_matrices.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Confusion matrices saved.")


def plot_roc_curves(models, X_test, y_test):
    """Generate ROC curves for all models."""

    plt.figure(figsize=(10, 7))

    for name, model in models.items():

        if hasattr(model, "decision_function"):
            scores = model.decision_function(X_test)
        else:
            scores = model.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, scores)
        roc_auc = auc(fpr, tpr)

        plt.plot(
            fpr,
            tpr,
            label=f"{name} (AUC = {roc_auc:.3f})"
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random classifier"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.savefig(
        "results/figures/roc_curves.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("ROC curves saved.")


def plot_model_comparison(results):
    """Plot accuracy and F1 score for all models."""

    x = range(len(results))

    plt.figure(figsize=(12, 6))

    plt.bar(
        [i - 0.2 for i in x],
        results["Accuracy"],
        width=0.4,
        label="Accuracy"
    )

    plt.bar(
        [i + 0.2 for i in x],
        results["F1 Score"],
        width=0.4,
        label="F1 Score"
    )

    plt.xticks(
        list(x),
        results["Model"],
        rotation=45,
        ha="right"
    )

    plt.ylabel("Score")
    plt.title("Model Performance Comparison")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "results/figures/model_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Model comparison plot saved.")