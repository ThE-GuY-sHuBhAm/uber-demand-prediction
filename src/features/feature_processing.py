import logging
from pathlib import Path
import pandas as pd

# create a logger
logger = logging.getLogger("feature_processing")
logger.setLevel(logging.INFO)

# attach a console handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)

# make a formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

if __name__ == "__main__":
    # current path
    current_path = Path(__file__)
    # set the root path
    root_path = current_path.parent.parent.parent
    # data_path
    data_path = root_path / "data/processed/resampled_data.csv"
    
    # read the data
    df = pd.read_csv(data_path, parse_dates=["tpep_pickup_datetime"])
    logger.info("Data read successfully")
    
    # --- NEW TIME FEATURES ---
    # extract the day of week information
    df["day_of_week"] = df["tpep_pickup_datetime"].dt.day_of_week
    # extract the month information
    df["month"] = df["tpep_pickup_datetime"].dt.month
    # extract the HOUR information (CRITICAL)
    df["hour"] = df["tpep_pickup_datetime"].dt.hour
    # extract weekend flag
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    
    logger.info("Datetime Features extracted successfully")
    
    # set the datetime column as index
    df.set_index("tpep_pickup_datetime", inplace=True)
    logger.info("Datetime column set as index successfully")
    
    # create the region grouper
    region_grp = df.groupby("region")
    
    # --- LAGS + ROLLING FEATURES ---
    
    feature_list = []
    
    # 1. Generate Lag Features (1 to 4 steps)
    for p in range(1, 5):
        lag_series = region_grp["total_pickups"].shift(p)
        lag_series.name = f"lag_{p}"
        feature_list.append(lag_series)

    # 2. Generate Rolling Mean (Window = 4 steps = 1 hour)
    # This captures the "trend" of the last hour
    rolling_series = region_grp["total_pickups"].shift(1).rolling(window=4).mean()
    rolling_series.name = "rolling_mean_4"
    feature_list.append(rolling_series)
        
    logger.info("Lag and Rolling features generated successfully")
    
    # merge them with the original df
    data = pd.concat(feature_list + [df], axis=1)
    
    logger.info("Features merged successfully")
    
    # drop the missing values (caused by shifting/rolling)
    data.dropna(inplace=True)
    
    # Define the final feature set
    features = [f"lag_{i}" for i in range(1, 5)] + \
               ["rolling_mean_4", "hour", "day_of_week", "is_weekend", "total_pickups", "region"]
    
    # split the data into train and test
    trainset = data.loc[data["month"].isin([1,2]), features]
    testset = data.loc[data["month"].isin([3]), features]
    
    # save the train and test data
    train_data_save_path = root_path / "data/processed/train.csv"
    test_data_save_path = root_path / "data/processed/test.csv"

    trainset.to_csv(train_data_save_path, index=True)
    logger.info("Train data saved successfully")
    
    testset.to_csv(test_data_save_path, index=True)
    logger.info("Test data saved successfully")