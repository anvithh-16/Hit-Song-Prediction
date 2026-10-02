from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from src.preprocessing import create_preprocessor


def build_models():
    """Create all machine learning models using the common preprocessor."""

    models = {

        "Logistic Regression": Pipeline([
            ("preprocessor", create_preprocessor()),
            ("model", LogisticRegression(
                max_iter=2000,
                solver="lbfgs"
            ))
        ]),

        "LDA": Pipeline([
            ("preprocessor", create_preprocessor()),
            ("model", LinearDiscriminantAnalysis(
                solver="lsqr",
                shrinkage="auto"
            ))
        ]),

        "Linear SVM": Pipeline([
            ("preprocessor", create_preprocessor()),
            ("model", SVC(
                kernel="linear"
            ))
        ]),

        "RBF SVM": Pipeline([
            ("preprocessor", create_preprocessor()),
            ("model", SVC(
                kernel="rbf"
            ))
        ]),

        "Polynomial SVM": Pipeline([
            ("preprocessor", create_preprocessor()),
            ("model", SVC(
                kernel="poly",
                degree=3
            ))
        ]),

        "Random Forest": Pipeline([
            ("preprocessor", create_preprocessor()),
            ("model", RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=-1
            ))
        ]),

        "Gradient Boosting": Pipeline([
            ("preprocessor", create_preprocessor()),
            ("model", GradientBoostingClassifier(
                n_estimators=100,
                random_state=42
            ))
        ]),

        "Neural Network": Pipeline([
            ("preprocessor", create_preprocessor()),
            ("model", MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=100,
                early_stopping=True,
                random_state=42
            ))
        ])
    }

    return models