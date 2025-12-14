import pandas as pd
import joblib
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score

# Constants
MODEL_DIR = "models"
DATA_PATH = "data/processed/data.csv"


def main():
    if not os.path.exists(DATA_PATH):
        print(f"Data not found at {DATA_PATH}. Run pipeline first.")
        return

    # Load & Prep
    df = pd.read_csv(DATA_PATH, index_col=0)  # Index=CustomerId
    X = df.drop(columns=["Risk_Label", "Cluster"])
    y = df["Risk_Label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")

    # Training Config
    models = {
        "LogisticReg": (LogisticRegression(random_state=42), {"C": [0.1, 1, 10]}),
        "GradientBoost": (
            GradientBoostingClassifier(random_state=42),
            {"n_estimators": [50, 100], "max_depth": [3]},
        ),
    }

    mlflow.set_experiment("Credit_Risk_Simple")
    best_auc = 0
    best_model = None

    for name, (model, params) in models.items():
        with mlflow.start_run(run_name=name):
            grid = GridSearchCV(model, params, cv=3, scoring="roc_auc")
            grid.fit(X_train_scaled, y_train)

            # Eval
            preds = grid.best_estimator_.predict(X_test_scaled)
            probs = grid.best_estimator_.predict_proba(X_test_scaled)[:, 1]
            auc = roc_auc_score(y_test, probs)
            f1 = f1_score(y_test, preds)

            print(f"{name} -> AUC: {auc:.4f}, F1: {f1:.4f}")

            mlflow.log_metrics({"auc": auc, "f1": f1})
            mlflow.log_params(grid.best_params_)

            if auc > best_auc:
                best_auc = auc
                best_model = grid.best_estimator_

    # Save Best
    joblib.dump(best_model, f"{MODEL_DIR}/best_model.pkl")
    print(f"Best model saved with AUC: {best_auc:.4f}")


if __name__ == "__main__":
    main()
