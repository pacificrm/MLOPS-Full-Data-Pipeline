import pandas as pd
import os

# --- Configuration: Adjust these variables ---
# 1. Path to your full, raw CSV file
input_csv_path = 'dataset/raw/transactions.csv'

# 2. Name of the column that contains the 0s and 1s
target_column = 'Class'

# 3. Output directory and file name
output_dir = 'sample'
output_csv_path = os.path.join(output_dir, 'sample.csv')

# --- Script Logic ---

try:
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the entire dataset
    print(f"Reading data from '{input_csv_path}'...")
    df = pd.read_csv(input_csv_path)

    # Separate the DataFrame by the target class
    df_class_0 = df[df[target_column] == 0]
    df_class_1 = df[df[target_column] == 1]

    # Take 3 random samples from each class
    sample_class_0 = df_class_0.sample(n=3, random_state=42)
    sample_class_1 = df_class_1.sample(n=3, random_state=42)

    # Combine the two samples into one DataFrame
    final_sample = pd.concat([sample_class_0, sample_class_1])

    # Shuffle the combined DataFrame to mix the rows
    final_sample = final_sample.sample(frac=1).reset_index(drop=True)

    # Save the final sample to the new CSV file
    final_sample.to_csv(output_csv_path, index=False)

    print(f"✅ Successfully created sample file at '{output_csv_path}'")
    print("\nSampled data distribution:")
    print(final_sample[target_column].value_counts())

except FileNotFoundError:
    print(f"❌ ERROR: The input file was not found at '{input_csv_path}'")
except Exception as e:
    print(f"An error occurred: {e}")
