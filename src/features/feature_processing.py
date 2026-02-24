import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger("feature_processing")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

if __name__ == "__main__":

    root = Path(__file__).parents[2]
    data_path = root / "data/processed/resampled_data.csv"

    df = pd.read_csv(data_path, parse_dates=["tpep_pickup_datetime"])
    logger.info("Resampled data loaded")

    # ---- SORT STRICTLY (CRITICAL) ----
    df = df.sort_values(["region", "tpep_pickup_datetime"])

    # ---- TIME FEATURES ----
    df["hour"] = df["tpep_pickup_datetime"].dt.hour
    df["day_of_week"] = df["tpep_pickup_datetime"].dt.day_of_week
    df["month"] = df["tpep_pickup_datetime"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    df = df.set_index("tpep_pickup_datetime")

    region_grp = df.groupby("region")

    # ---- LAG FEATURES ----
    for p in range(1, 5):
        df[f"lag_{p}"] = region_grp["total_pickups"].shift(p)

    # ---- ROLLING TREND ----
    df["rolling_mean_4"] = (
        region_grp["total_pickups"]
        .shift(1)
        .rolling(4)
        .mean()
    )

    # ---- TARGET = NEXT STEP DEMAND ----
    df["target"] = region_grp["total_pickups"].shift(-1)

    # ---- DROP INVALID ROWS ----
    df.dropna(inplace=True)

    features = [
        "lag_1", "lag_2", "lag_3", "lag_4",
        "rolling_mean_4",
        "hour", "day_of_week", "is_weekend",
        "region"
    ]

    df = df.sort_index()

    # ---- TEMPORAL SPLIT (ONCE, CLEAN) ----
    train = df[df["month"].isin([1, 2])]
    test = df[df["month"] == 3]

    X_train = train[features]
    y_train = train["target"]

    X_test = test[features]
    y_test = test["target"]

    train_out = pd.concat([X_train, y_train], axis=1)
    test_out = pd.concat([X_test, y_test], axis=1)

    train_out.to_csv(root / "data/processed/train.csv")
    test_out.to_csv(root / "data/processed/test.csv")

    logger.info("Leakage-free train/test datasets created")