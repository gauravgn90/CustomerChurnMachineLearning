import numpy as np
import pandas as pd

SERVICE_COLS = [
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies",
]
NOT_SUBSCRIBED = ["No", "No phone service", "No internet service"]


def add_features(X):
    """Get Raw customer data add 3 new columns to it and return"""
    X = X.copy()

    # 1. No of services consumed by customer. Count of all not subscribed services
    X["TotalServices"] = sum(
        (~X[c].isin(NOT_SUBSCRIBED)).astype(int) for c in SERVICE_COLS
    )

    # 2. Charges per service consumed by customer. MonthlyCharges / TotalServices. 
    # We use clip(lower=1) to avoid division by zero in case TotalServices is 0.
    X["ChargePerService"] = X["MonthlyCharges"] / X["TotalServices"].clip(lower=1)

    # 3. Tenure group of customer. We create bins for tenure and label them accordingly.
    # Use pd.cut to create bins for tenure and label them accordingly. The bins are defined as follows:
    # -1 to 6 months: "0-6"
    # 7 to 12 months: "7-12"
    # 13 to 24 months: "13-24"
    # 25 to 48 months: "25-48"
    # 49 to 72 months: "49-72"
    
    X["TenureGroup"] = pd.cut(
        X["tenure"], bins=[-1, 6, 12, 24, 48, 72],
        labels=["0-6", "7-12", "13-24", "25-48", "49-72"],
    ).astype(str)

    return X