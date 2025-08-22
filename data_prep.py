import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

# Create directories
os.makedirs('data/processed', exist_ok=True)
os.makedirs('data/poisoned', exist_ok=True)

df = pd.read_csv('data/raw/transactions.csv')
df = df.sort_values('Time').reset_index(drop=True)

# Split into two versions
midpoint = len(df) // 2
df_v1 = df.iloc[:midpoint]
df_v2 = df.iloc[midpoint:]

df_v1.to_parquet('data/processed/transactions_v1.parquet', index=False)
df_v2.to_parquet('data/processed/transactions_v2.parquet', index=False)

# Create poisoned datasets from version 1
def poison_data(df, percentage):
    poisoned_df = df.copy()
    n_samples_to_flip = int(len(poisoned_df) * (percentage / 100))
    indices_to_flip = np.random.choice(poisoned_df.index, n_samples_to_flip, replace=False)
    poisoned_df.loc[indices_to_flip, 'Class'] = 1 - poisoned_df.loc[indices_to_flip, 'Class']
    return poisoned_df

for p in [2, 8, 15, 30]:
    poisoned_df = poison_data(df_v1, p)
    poisoned_df.to_parquet(f'data/poisoned/transactions_v1_poisoned_{p}percent.parquet', index=False)
