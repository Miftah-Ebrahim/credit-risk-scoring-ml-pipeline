import joblib
import pandas as pd
import os

MODEL_PATH = "models/best_model.pkl"
SCALER_PATH = "models/scaler.pkl"

_model = None
_scaler = None


def load_artifacts():
    """Loads model/scaler into global variables."""
    global _model, _scaler
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        _model = joblib.load(MODEL_PATH)
        _scaler = joblib.load(SCALER_PATH)
        return True
    return False


def predict_risk(data: dict):
    """
    Predicts risk for a single record.
    Data format: {Recency, Frequency, Monetary_Total, ...}
    """
    if _model is None:
        if not load_artifacts():
            raise RuntimeError("Model artifacts not found.")

    # Prepare
    df = pd.DataFrame([data])
    # Ensure column order matches training (Recency, Frequency, Monetary_Total, Monetary_Mean, Monetary_Std)
    # Note: In production, we should enforce schema strictly.
    cols = ["Recency", "Frequency", "Monetary_Total", "Monetary_Mean", "Monetary_Std"]
    df = df[cols]

    # Scale
    X_scaled = _scaler.transform(df)

    # Predict
    prob = _model.predict_proba(X_scaled)[0][1]
    is_risk = int(_model.predict(X_scaled)[0])

    return {"risk_probability": round(prob, 4), "is_high_risk": bool(is_risk)}
