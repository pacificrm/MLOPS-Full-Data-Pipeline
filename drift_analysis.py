import pandas as pd
import mlflow
import json
# Updated imports as per your request
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset
import os

# --- Configuration ---
# Reference data (what the model was trained on)
REFERENCE_FILE = 'data/train.parquet'
# Current data (the new data we want to check for drift)
CURRENT_FILE = 'dataset/processed/transactions_v2.parquet'
TARGET_COLUMN = 'Class'

# --- MLflow Configuration ---
mlflow.set_tracking_uri("http://127.0.0.1:8000")
EXPERIMENT_NAME = "Fraud Detection - Data Drift Analysis"
mlflow.set_experiment(EXPERIMENT_NAME)

def run_drift_analysis():
    """
    Performs data and target drift analysis and logs reports to MLflow.
    """
    print("--- Starting Data Drift Analysis ---")

    # 1. Load Data
    print("Loading reference and current datasets...")
    try:
        reference_df = pd.read_parquet(REFERENCE_FILE)
        current_df = pd.read_parquet(CURRENT_FILE)
    except FileNotFoundError as e:
        print(f"Error: Data file not found. Please ensure files exist.")
        print(e)
        return

    with mlflow.start_run(run_name="Drift_Analysis_V1_vs_V2"):
        # 2. Perform Data Drift and Summary Analysis
        print("Generating Data Drift and Summary report...")
        # Create the report configuration
        data_drift_report_config = Report(metrics=[
            DataDriftPreset(),
            DataSummaryPreset()
        ])
        
        # Capture the object returned by the .run() method
        report_result = data_drift_report_config.run(reference_data=reference_df, current_data=current_df)

        # Use the returned object to get the HTML content
        html_content = report_result._repr_html_()
        mlflow.log_text(html_content, "reports/data_drift_and_summary_report.html")
        print("  -> Data drift and summary report logged directly to MLflow.")

        # 3. Extract and log key drift metrics to MLflow for easy comparison
        print("Extracting and logging key drift metrics...")
        # Use .dict() to get the report results as a dictionary
        drift_results = report_result.dict()

        # Save the full JSON report locally for inspection, as requested
        with open("my_eval.json", "w") as f:
            json.dump(drift_results, f, indent=4)
        print("  -> Saved full drift report to my_eval.json")
        mlflow.log_artifact("my_eval.json")

        # CORRECTED: Access the correct key for the number of drifted features
        # The first metric in the list is 'DriftedColumnsCount'
        num_drifted_features = drift_results['metrics'][0]['value']['count']
        mlflow.log_metric("num_drifted_features", num_drifted_features)
        print(f"  -> Logged 'num_drifted_features': {num_drifted_features}")

    print("\n✅ Drift analysis experiment complete.")
    print("📊 Check MLflow UI for the full HTML reports and metrics.")

if __name__ == "__main__":
    run_drift_analysis()
