import pandas as pd
import shutil
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score


def main():
    data = pd.read_csv("data/processed/data.csv", index_col=0)
    X = data.drop(columns=["Risk_Label", "Cluster"])
    y = data["Risk_Label"]

    X["ChannelId"] = X["ChannelId"].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    prep = ColumnTransformer(
        [
            (
                "num",
                StandardScaler(),
                [
                    "Recency",
                    "Frequency",
                    "Monetary_Total",
                    "Monetary_Mean",
                    "Monetary_Std",
                ],
            ),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["ChannelId"]),
        ]
    )

    mlflow.set_experiment("Credit_Risk_Model")

    models = {
        "LogReg": (LogisticRegression(), {"model__C": [0.1, 1, 10]}),
        "GBM": (GradientBoostingClassifier(), {"model__n_estimators": [50, 100]}),
    }

    best_score, best_model = 0, None

    for name, (clf, params) in models.items():
        with mlflow.start_run(run_name=name):
            pipe = Pipeline([("prep", prep), ("model", clf)])
            grid = GridSearchCV(pipe, params, cv=3, scoring="roc_auc").fit(
                X_train, y_train
            )

            auc = roc_auc_score(y_test, grid.predict_proba(X_test)[:, 1])
            mlflow.log_metric("auc", auc)
            mlflow.log_params(grid.best_params_)

            if auc > best_score:
                best_score, best_model = auc, grid.best_estimator_

    shutil.rmtree("models/production_model", ignore_errors=True)
    mlflow.sklearn.save_model(best_model, "models/production_model")
    print(f"Best AUC: {best_score:.4f}")


if __name__ == "__main__":
    main()
