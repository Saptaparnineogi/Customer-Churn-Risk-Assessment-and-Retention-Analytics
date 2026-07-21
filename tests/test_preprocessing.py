import pandas as pd

from src.feature_engineering import engineer_features
from src.preprocessing import get_feature_types


def test_total_charges_is_numeric():
    df = pd.DataFrame(
        {
            "gender": ["Female", "Male"],
            "SeniorCitizen": [0, 1],
            "Partner": ["Yes", "No"],
            "Dependents": ["No", "No"],
            "tenure": [1, 12],
            "PhoneService": ["No", "Yes"],
            "MultipleLines": ["No phone service", "No"],
            "InternetService": ["DSL", "Fiber optic"],
            "OnlineSecurity": ["No", "No"],
            "OnlineBackup": ["Yes", "No"],
            "DeviceProtection": ["No", "Yes"],
            "TechSupport": ["No", "No"],
            "StreamingTV": ["No", "Yes"],
            "StreamingMovies": ["No", "Yes"],
            "Contract": ["Month-to-month", "One year"],
            "PaperlessBilling": ["Yes", "No"],
            "PaymentMethod": [
                "Electronic check",
                "Mailed check",
            ],
            "MonthlyCharges": [29.85, 56.95],
            "TotalCharges": [29.85, 684.40],
        }
    )

    df = engineer_features(df)

    numerical_features, categorical_features = get_feature_types(df)

    assert "TotalCharges" in numerical_features
    assert "TotalCharges" not in categorical_features