import pandas as pd
import mlflow.pyfunc

MODEL_PATH = "models/production_model"
_model = None


def load_model():
    global _model
    try:
        _model = mlflow.pyfunc.load_model(MODEL_PATH)
    except:
        pass


def predict_risk(data: dict):
    if not _model:
        load_model()
    if not _model:
        raise RuntimeError("Model unavailable")

    df = pd.DataFrame([data])
    if "ChannelId" in df:
        df["ChannelId"] = df["ChannelId"].astype(str)

    try:
        est = _model._model_impl
        prob = est.predict_proba(df)[0][1]
        pred = int(est.predict(df)[0])
    except:
        pred = int(_model.predict(df)[0])
        prob = float(pred)

    return {"risk_probability": round(prob, 4), "is_high_risk": bool(pred)}
