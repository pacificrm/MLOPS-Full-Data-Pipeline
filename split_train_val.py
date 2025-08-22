import pandas as pd
from sklearn.model_selection import train_test_split
import os

# --- Configuration ---
INPUT_FILE = 'data/transactions_v1.parquet'
OUTPUT_DIR = 'data/'
TRAIN_FILE = os.path.join(OUTPUT_DIR, 'train.parquet')
VAL_FILE = os.path.join(OUTPUT_DIR, 'val.parquet')
TEST_SIZE = 0.2
RANDOM_STATE = 42
TARGET_COLUMN = 'Class'

def split_data():
    """Reads the input data, splits it, and saves train/val sets."""
    print(f"Reading data from {INPUT_FILE}...")
    try:
        df = pd.read_parquet(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{INPUT_FILE}'")
        return

    # Ensure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Splitting data into train and validation sets (test_size={TEST_SIZE})...")

    # Perform a stratified split to maintain the target distribution
    train_df, val_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[TARGET_COLUMN]
    )

    print(f"Train set shape: {train_df.shape}")
    print(f"Validation set shape: {val_df.shape}")

    # Save the splits
    train_df.to_parquet(TRAIN_FILE, index=False)
    val_df.to_parquet(VAL_FILE, index=False)

    print(f"✅ Train data saved to: {TRAIN_FILE}")
    print(f"✅ Validation data saved to: {VAL_FILE}")

if __name__ == "__main__":
    split_data()

