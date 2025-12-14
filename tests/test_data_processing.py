import pytest
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features import (
    calculate_rfm,
    assign_risk_label,
    add_temporal_features,
    calculate_woe_iv,
)


@pytest.fixture
def sample_data():
    data = {
        "TransactionId": ["T1", "T2", "T3"],
        "CustomerId": ["C1", "C1", "C2"],
        "Amount": [100.0, 200.0, 500.0],
        "TransactionStartTime": pd.to_datetime(
            ["2023-01-01 10:00:00", "2023-01-02 12:30:00", "2023-01-05 15:45:00"]
        ),
    }
    return pd.DataFrame(data)


def test_calculate_rfm(sample_data):
    res = calculate_rfm(sample_data)
    assert res.shape == (2, 5)
    assert "Recency" in res.columns
    # C1 total amount = 300
    assert res.loc["C1", "Monetary_Total"] == 300.0


def test_assign_risk_label(sample_data):
    rfm = calculate_rfm(sample_data)
    res = assign_risk_label(rfm, n_clusters=2)
    assert "Risk_Label" in res.columns
    assert res["Risk_Label"].isin([0, 1]).all()


def test_add_temporal_features(sample_data):
    res = add_temporal_features(sample_data)
    assert "TransactionHour" in res.columns
    assert "TransactionDay" in res.columns
    assert "TransactionMonth" in res.columns
    assert res["TransactionHour"].iloc[0] == 10


def test_calculate_woe_iv():
    # Simple Mock Data for WoE
    data = pd.DataFrame(
        {"Feature_Bin": ["A", "A", "B", "B", "A"], "Target": [0, 1, 0, 0, 0]}
    )

    res = calculate_woe_iv(data, "Feature_Bin", "Target")
    assert "IV" in res
    assert "WoE_Table" in res
    assert res["IV"] >= 0
