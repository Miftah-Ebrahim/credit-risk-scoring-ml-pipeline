import pandas as pd
import numpy as np
import os
import logging
from src.features import (
    calculate_rfm,
    assign_risk_label,
    add_temporal_features,
    calculate_woe_iv,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")


def process_data(input_path: str, output_path: str):
    if not os.path.exists(input_path):
        return

    df = pd.read_csv(input_path)
    if "TransactionStartTime" in df:
        df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])

    df = add_temporal_features(df.drop_duplicates())
    df.fillna(df.median(numeric_only=True), inplace=True)

    rfm = calculate_rfm(df)
    final = assign_risk_label(rfm)

    for feat in ["Recency", "Frequency", "Monetary_Total"]:
        try:
            temp = final.copy()
            temp[f"{feat}_Bin"] = pd.qcut(temp[feat], q=4, duplicates="drop")
            iv = calculate_woe_iv(temp, f"{feat}_Bin", "Risk_Label")["IV"]
            logging.info(f"{feat} IV: {iv:.4f}")
        except:
            pass

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final.to_csv(output_path)
    logging.info(f"Processed data saved to {output_path}")


if __name__ == "__main__":
    raw = "data/raw"
    files = [f for f in os.listdir(raw) if f.endswith(".csv")]
    if files:
        process_data(os.path.join(raw, files[0]), "data/processed/data.csv")
