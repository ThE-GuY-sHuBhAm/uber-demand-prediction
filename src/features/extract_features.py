import joblib
import pandas as pd
import logging
from pathlib import Path
from yaml import safe_load
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("extract_features")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

def read_cluster_input(data_path, chunksize=100000):
    return pd.read_csv(
        data_path,
        chunksize=chunksize,
        usecols=["pickup_latitude", "pickup_longitude"]
    )

def read_params():
    with open("params.yaml") as f:
        return safe_load(f)

if __name__ == "__main__":
    root = Path(__file__).parents[2]

    raw_full_path = root / "data/interim/df_without_outliers.csv"
    resampled_path = root / "data/processed/resampled_data.csv"

    # ---- TRAIN CLUSTERING MODELS (USING FULL DATA UP TO TRAIN PERIOD) ----
    # We still fit on early data only to stay realistic

    train_raw = root / "data/interim/train_raw.csv"

    scaler = StandardScaler()
    for chunk in read_cluster_input(train_raw):
        scaler.partial_fit(chunk)

    joblib.dump(scaler, root / "models/scaler.joblib")

    kmeans_params = read_params()["extract_features"]["mini_batch_kmeans"]
    kmeans = MiniBatchKMeans(**kmeans_params)

    for chunk in read_cluster_input(train_raw):
        kmeans.partial_fit(scaler.transform(chunk))

    joblib.dump(kmeans, root / "models/mb_kmeans.joblib")

    logger.info("Spatial models trained")

    # ---- APPLY CLUSTERING TO FULL DATA ----

    df = pd.read_csv(raw_full_path, parse_dates=["tpep_pickup_datetime"])

    coords = df[["pickup_longitude", "pickup_latitude"]]
    df["region"] = kmeans.predict(scaler.transform(coords))

    df = df.drop(columns=["pickup_longitude", "pickup_latitude"])

    df = df.set_index("tpep_pickup_datetime")

    resampled = (
        df.groupby("region")
          .resample("15min")
          .size()
          .rename("total_pickups")
          .reset_index(level=0)
    )

    resampled.replace(0, 1, inplace=True)

    resampled.to_csv(resampled_path)
    logger.info("Resampled demand timeline saved cleanly")