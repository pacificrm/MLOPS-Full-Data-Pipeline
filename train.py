import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mlflow.models.signature import infer_signature
from tqdm import tqdm
import os

# --- Configuration ---
TRAIN_FILE = 'data/train.parquet'
VAL_FILE = 'data/val.parquet'
MODEL_OUTPUT_PATH = 'model.joblib'
METRICS_OUTPUT_PATH = 'metrics.txt'
TARGET_COLUMN = 'Class'
RANDOM_STATE = 42

# --- MLflow Configuration ---
# Make sure your MLflow tracking server is running, e.g., `mlflow ui`
# If running locally, the default URI is often 'http://127.0.0.1:5000'
# Update this if your server is elsewhere.
mlflow.set_tracking_uri("http://127.0.0.1:8000")
mlflow.set_experiment("Fraud Detection")

def train_model():
    """Loads data, trains a model, evaluates, and logs everything to MLflow."""
    print("--- Starting Model Training & Logging ---")

    # 1. Load Data
    print("Loading training and validation data...")
    try:
        train_df = pd.read_parquet(TRAIN_FILE)
        val_df = pd.read_parquet(VAL_FILE)
    except FileNotFoundError as e:
        print(f"Error: Data file not found. Please run split_data.py first.")
        print(e)
        return

    # 2. Prepare Data
    features = [col for col in train_df.columns if col not in [TARGET_COLUMN, 'Time']]
    X_train = train_df[features]
    y_train = train_df[TARGET_COLUMN]
    X_val = val_df[features]
    y_val = val_df[TARGET_COLUMN]

    print(f"Training with {len(features)} features.")

    # Start an MLflow run
    with mlflow.start_run(run_name="DecisionTree_Initial_Run"):
        # 3. Train Model
        print("Training DecisionTreeClassifier...")
        model = DecisionTreeClassifier(random_state=RANDOM_STATE)

        with tqdm(total=1, desc="Fitting model") as pbar:
            model.fit(X_train, y_train)
            pbar.update(1)

        print("✅ Model training complete.")

        # 4. Evaluate Model
        print("Evaluating model on the validation set...")
        y_pred = model.predict(X_val)

        metrics = {
            "accuracy": accuracy_score(y_val, y_pred),
            "precision": precision_score(y_val, y_pred),
            "recall": recall_score(y_val, y_pred),
            "f1_score": f1_score(y_val, y_pred)
        }

        print("Evaluation Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        # --- 5. Logging and Saving ---
        print("Logging to MLflow and saving local artifacts...")

        # Log parameters to MLflow
        mlflow.log_params({
            "model_type": "DecisionTreeClassifier",
            "random_state": RANDOM_STATE
        })

        # Log metrics to MLflow
        mlflow.log_metrics(metrics)

        # Create a model signature and an input example
        signature = infer_signature(X_train, model.predict(X_train))
        input_example = X_train.head(5)

        # Log the model to MLflow
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="fraud_detector_model",
            signature=signature,
            input_example=input_example
        )
        print("  -> Model, params, and metrics logged to MLflow.")

        # Save the model locally
        joblib.dump(model, MODEL_OUTPUT_PATH)
        print(f"  -> Local model saved to: {MODEL_OUTPUT_PATH}")

        # Save the metrics locally
        with open(METRICS_OUTPUT_PATH, 'w') as f:
            for k, v in metrics.items():
                f.write(f"{k}: {v:.4f}\n")
        print(f"  -> Local metrics saved to: {METRICS_OUTPUT_PATH}")

    print("\n--- Script Finished ---")


if __name__ == "__main__":
    train_model()
