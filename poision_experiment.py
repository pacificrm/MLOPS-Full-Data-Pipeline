import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from mlflow.models.signature import infer_signature
from tqdm import tqdm
import os

# --- Configuration ---
TRAIN_FILE = 'data/train.parquet'
VAL_FILE = 'data/val.parquet'
TARGET_COLUMN = 'Class'
RANDOM_STATE = 42
POISON_LEVELS = [0, 2, 8, 15, 30] # 0 is the baseline

# --- MLflow Configuration ---
# Ensure your MLflow server is running (e.g., `mlflow ui`)
mlflow.set_tracking_uri("http://127.0.0.1:8000")
EXPERIMENT_NAME = "Fraud Detection - Poisoning Impact"
mlflow.set_experiment(EXPERIMENT_NAME)

def poison_data(df, target_column, percentage):
    """
    Flips the labels of a given percentage of samples in the dataframe.
    """
    if percentage == 0:
        return df

    poisoned_df = df.copy()
    n_samples_to_flip = int(len(poisoned_df) * (percentage / 100))
    
    # Randomly select indices to flip the target label
    indices_to_flip = np.random.choice(poisoned_df.index, n_samples_to_flip, replace=False)
    
    # Flip the labels (0 becomes 1, 1 becomes 0)
    poisoned_df.loc[indices_to_flip, target_column] = 1 - poisoned_df.loc[indices_to_flip, target_column]
    
    print(f"Poisoned {n_samples_to_flip} samples ({percentage}%)")
    return poisoned_df

def run_poisoning_experiment():
    """
    Runs the full data poisoning experiment and logs results to MLflow.
    """
    print("--- Starting Data Poisoning Impact Analysis ---")

    # 1. Load clean data
    print("Loading clean training and validation data...")
    try:
        train_df = pd.read_parquet(TRAIN_FILE)
        val_df = pd.read_parquet(VAL_FILE)
    except FileNotFoundError as e:
        print(f"Error: Data file not found. Please ensure '{TRAIN_FILE}' and '{VAL_FILE}' exist.")
        print(e)
        return

    # Prepare clean validation data (this never changes)
    features = [col for col in val_df.columns if col not in [TARGET_COLUMN, 'Time']]
    X_val = val_df[features]
    y_val = val_df[TARGET_COLUMN]

    # 2. Loop through each poisoning level
    for p_level in POISON_LEVELS:
        run_name = f"baseline" if p_level == 0 else f"poisoned_{p_level}_percent"
        
        with mlflow.start_run(run_name=run_name):
            print(f"\n--- Running experiment for poisoning level: {p_level}% ---")
            mlflow.log_param("poison_percentage", p_level)

            # a. Create poisoned training data
            poisoned_train_df = poison_data(train_df, TARGET_COLUMN, p_level)
            X_train_poisoned = poisoned_train_df[features]
            y_train_poisoned = poisoned_train_df[TARGET_COLUMN]

            # b. Train model on (potentially) poisoned data
            model = DecisionTreeClassifier(random_state=RANDOM_STATE)
            
            # Added tqdm for a visual progress bar during training
            with tqdm(total=1, desc=f"Training model ({p_level}% poison)") as pbar:
                model.fit(X_train_poisoned, y_train_poisoned)
                pbar.update(1)

            # c. Evaluate on the CLEAN validation data
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)[:, 1]

            metrics = {
                "accuracy": accuracy_score(y_val, y_pred),
                "precision": precision_score(y_val, y_pred),
                "recall": recall_score(y_val, y_pred),
                "f1_score": f1_score(y_val, y_pred),
                "roc_auc": roc_auc_score(y_val, y_pred_proba)
            }

            print("Evaluation Metrics:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")

            # d. Log everything to MLflow
            mlflow.log_metrics(metrics)
            
            # Create a model signature and input example to resolve warnings
            signature = infer_signature(X_train_poisoned, model.predict(X_train_poisoned))
            input_example = X_train_poisoned.head(5)

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=f"model_poisoned_{p_level}",
                signature=signature,
                input_example=input_example
            )
            print("Logged metrics and model to MLflow.")

    # 3. Create and log a summary report
    print("\n--- Generating Performance Degradation Report ---")
    
    # Fetch all runs from this experiment
    runs = mlflow.search_runs(experiment_names=[EXPERIMENT_NAME])
    
    # Filter and sort the results
    report_df = runs[["params.poison_percentage", "metrics.f1_score", "metrics.accuracy", "metrics.roc_auc", "metrics.recall", "metrics.precision"]].copy()
    report_df["params.poison_percentage"] = pd.to_numeric(report_df["params.poison_percentage"])
    report_df = report_df.sort_values("params.poison_percentage").reset_index(drop=True)
    
    print("Performance Summary:")
    print(report_df)

    # Save report to a file
    report_path = "poisoning_degradation_report.md"
    report_df.to_markdown(report_path, index=False)

    # Log the report as an artifact to the last run
    with mlflow.start_run(run_id=runs.iloc[0].run_id):
         mlflow.log_artifact(report_path, "summary_reports")

    print(f"\n✅ Experiment complete. Summary report saved and logged to MLflow.")

if __name__ == "__main__":
    run_poisoning_experiment()
