import joblib
import pandas as pd
import logging
from pathlib import Path
from yaml import safe_load
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

# create a logger
logger = logging.getLogger("extract_features")
logger.setLevel(logging.INFO)

# attach a console handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)

# make a formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

def read_cluster_input(data_path, chunksize=100000, usecols=["pickup_latitude","pickup_longitude"]):
    df_reader = pd.read_csv(data_path, chunksize=chunksize, usecols=usecols)
    return df_reader

def save_model(model, save_path):
    joblib.dump(model, save_path)

def read_params(params_path="params.yaml"):
    with open(params_path, "r") as file:
        params = safe_load(file)
    return params

def process_data(data_path, scaler, kmeans_model, ewma_params):
    """
    Helper function to load data, predict regions, resample, and smooth.
    Used for both Train and Test sets to ensure consistent processing.
    """
    logger.info(f"Processing data from: {data_path}")
    
    # Read full data for this split
    df = pd.read_csv(data_path, parse_dates=["tpep_pickup_datetime"])
    
    # Select coordinates for prediction
    location_subset = df.loc[:, ["pickup_longitude", "pickup_latitude"]]
    
    # Scale inputs using the PRE-TRAINED scaler
    scaled_location_subset = scaler.transform(location_subset)
    
    # Predict clusters using the PRE-TRAINED kmeans
    cluster_predictions = kmeans_model.predict(scaled_location_subset)
    
    # Assign regions
    df['region'] = cluster_predictions
    
    # Drop coords
    df = df.drop(columns=["pickup_latitude", "pickup_longitude"])
    
    # Resample logic
    df.set_index('tpep_pickup_datetime', inplace=True)
    region_grp = df.groupby("region")
    
    resampled_data = (
        region_grp['region']
        .resample("15min")
        .count()
    )
    resampled_data.name = "total_pickups"
    
    # Reset index to get columns back
    resampled_data = resampled_data.reset_index(level=0)
    
    # Epsilon replacement
    epsilon_val = 1
    resampled_data.replace({'total_pickups': {0 : epsilon_val}}, inplace=True)
    
    # EWMA Smoothing
    resampled_data["avg_pickups"] = (
        resampled_data
        .groupby("region")['total_pickups']
        .ewm(**ewma_params)
        .mean()
        .round()
        .values
    )
    
    return resampled_data

if __name__ == "__main__":
    # current path
    current_path = Path(__file__)
    # set the root path
    root_path = current_path.parent.parent.parent
    
    # Paths for inputs
    train_data_path = root_path / "data/interim/train_raw.csv"
    test_data_path = root_path / "data/interim/test_raw.csv"
    
    # --- PHASE 1: TRAINING (Using ONLY Train Data) ---
    
    # read the data for clustering
    df_reader = read_cluster_input(train_data_path)
    logger.info("Train data read for model fitting")
    
    # train the standard scaler
    scaler = StandardScaler()
    for chunk in df_reader:
        scaler.partial_fit(chunk)
        
    # save the scaler
    scaler_save_path = root_path / "models/scaler.joblib"
    save_model(scaler, scaler_save_path)
    logger.info("Scaler trained and saved successfully")
    
    # read the parameters
    mini_batch_params = read_params()["extract_features"]["mini_batch_kmeans"]
    print("Parameters for clustering are ", mini_batch_params)
    
    # train the kmeans model
    mini_batch = MiniBatchKMeans(**mini_batch_params)
    
    # Reload reader for K-Means pass
    df_reader = read_cluster_input(train_data_path)
    for chunk in df_reader:
        scaled_chunk = scaler.transform(chunk)
        mini_batch.partial_fit(scaled_chunk)
        
    # save the model
    kmeans_save_path = root_path / "models/mb_kmeans.joblib"
    joblib.dump(mini_batch, kmeans_save_path)
    logger.info("K-Means trained and saved successfully")
    
    # --- PHASE 2: TRANSFORMATION (Process Both Train and Test) ---
    
    ewma_params = read_params()["extract_features"]["ewma"]
    
    # Process Train
    processed_train = process_data(train_data_path, scaler, mini_batch, ewma_params)
    logger.info("Train data processed successfully")
    
    # Process Test
    processed_test = process_data(test_data_path, scaler, mini_batch, ewma_params)
    logger.info("Test data processed successfully")
    
    # --- PHASE 3: MERGE AND SAVE ---
    # We concatenate them back because the next stage (feature_processing.py) 
    # expects a single file to generate lags and then splits by month again.
    
    final_df = pd.concat([processed_train, processed_test])
    
    # save the data
    save_path = root_path / "data/processed/resampled_data.csv"
    final_df.to_csv(save_path, index=True)
    logger.info("Combined resampled data saved successfully")