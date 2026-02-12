import pandas as pd
import logging
from pathlib import Path

# create a logger
logger = logging.getLogger("split_data")
logger.setLevel(logging.INFO)

# attach a console handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)

# make a formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

def split_and_save_data(input_path, train_path, test_path, chunksize=100000):
    """
    Reads the input data in chunks, splits based on month, 
    and appends to train/test CSVs incrementally.
    """
    
    # Make sure output files are empty before starting (or created if not exist)
    # We write headers only for the first chunk
    first_chunk = True
    
    # Create/Clear the files
    with open(train_path, 'w') as f: pass
    with open(test_path, 'w') as f: pass

    logger.info(f"Starting split process. Reading from {input_path}")
    
    # Iterate through chunks
    with pd.read_csv(input_path, chunksize=chunksize, parse_dates=["tpep_pickup_datetime"]) as reader:
        for i, chunk in enumerate(reader):
            
            # Create a month column for filtering
            chunk['month'] = chunk['tpep_pickup_datetime'].dt.month
            
            # Split logic: Jan (1) & Feb (2) -> Train, March (3) -> Test
            train_chunk = chunk[chunk['month'].isin([1, 2])].drop(columns=['month'])
            test_chunk = chunk[chunk['month'] == 3].drop(columns=['month'])
            
            # Append to CSVs
            train_chunk.to_csv(train_path, mode='a', header=first_chunk, index=False)
            test_chunk.to_csv(test_path, mode='a', header=first_chunk, index=False)
            
            first_chunk = False
            
            if i % 10 == 0:
                logger.info(f"Processed chunk {i}...")

    logger.info("Data split completed successfully.")

if __name__ == "__main__":
    # current path
    current_path = Path(__file__)
    # set the root path
    root_path = current_path.parent.parent.parent
    
    # Input path
    input_data_path = root_path / "data/interim/df_without_outliers.csv"
    
    # Output paths
    train_data_path = root_path / "data/interim/train_raw.csv"
    test_data_path = root_path / "data/interim/test_raw.csv"
    
    split_and_save_data(input_data_path, train_data_path, test_data_path)