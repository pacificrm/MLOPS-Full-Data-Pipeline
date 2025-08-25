import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import shap
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from fairlearn.metrics import MetricFrame, demographic_parity_difference, equalized_odds_difference, selection_rate
import matplotlib.pyplot as plt
from functools import partial
import os

# --- Configuration ---
TRAIN_FILE = 'data/train.parquet'
VAL_FILE = 'data/val.parquet'
TARGET_COLUMN = 'Class'
RANDOM_STATE = 42
SENSITIVE_COLUMN = 'location'

# --- MLflow Configuration ---
mlflow.set_tracking_uri("http://127.0.0.1:8000")
EXPERIMENT_NAME = "Fraud Detection - Fairness & Explainability"
mlflow.set_experiment(EXPERIMENT_NAME)

def create_fairness_plots(y_val, y_pred, y_pred_proba, sensitive_features_val, fairness_metrics):
    """
    Create comprehensive fairness visualization plots including MetricFrame analysis
    """
    os.makedirs("fairness_plots", exist_ok=True)
    
    # Define comprehensive metrics for MetricFrame analysis
    metrics = {
        'accuracy': accuracy_score,
        'precision': partial(precision_score, average='binary', zero_division=0),
        'recall': partial(recall_score, average='binary', zero_division=0),
        'selection_rate': selection_rate  # Added for comprehensive fairness analysis
    }
    
    # Create MetricFrame for comprehensive fairness assessment
    print("🔬 Starting comprehensive fairness assessment with MetricFrame...")
    grouped_metrics = MetricFrame(
        metrics=metrics,
        y_true=y_val,
        y_pred=y_pred,
        sensitive_features=sensitive_features_val
    )
    
    overall_metrics = grouped_metrics.overall
    group_metrics = grouped_metrics.by_group
    
    # 1. Create HTML report for fairness metrics
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Fairness Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .metric-table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            .metric-table th, .metric-table td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
            .metric-table th {{ background-color: #f2f2f2; }}
            .section {{ margin: 30px 0; }}
        </style>
    </head>
    <body>
        <h1>🔍 Comprehensive Fairness Analysis Report</h1>
        <div class="section">
            <h2>📊 Overall Performance</h2>
            <table class="metric-table">
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Accuracy</td><td>{overall_metrics['accuracy']:.4f}</td></tr>
                <tr><td>Precision</td><td>{overall_metrics['precision']:.4f}</td></tr>
                <tr><td>Recall</td><td>{overall_metrics['recall']:.4f}</td></tr>
                <tr><td>Selection Rate</td><td>{overall_metrics['selection_rate']:.4f}</td></tr>
            </table>
        </div>
        <div class="section">
            <h2>👥 Performance by Group</h2>
            <table class="metric-table">
                <tr><th>Group</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>Selection Rate</th></tr>
    """
    
    for group in group_metrics.index:
        html_content += f"""
                <tr>
                    <td>Group {group}</td>
                    <td>{group_metrics.loc[group, 'accuracy']:.4f}</td>
                    <td>{group_metrics.loc[group, 'precision']:.4f}</td>
                    <td>{group_metrics.loc[group, 'recall']:.4f}</td>
                    <td>{group_metrics.loc[group, 'selection_rate']:.4f}</td>
                </tr>
        """
    
    html_content += """
            </table>
        </div>
    </body>
    </html>
    """
    
    # Save HTML report
    html_report_path = "fairness_plots/fairness_analysis_report.html"
    with open(html_report_path, 'w') as f:
        f.write(html_content)
    
    return html_report_path


def run_fairness_experiment():
    """
    Runs a fairness and explainability analysis and logs results to MLflow.
    """
    print("--- Starting Fairness & Explainability Analysis ---")

    # 1. Load Data
    print("Loading training and validation data...")
    try:
        train_df = pd.read_parquet(TRAIN_FILE)
        val_df = pd.read_parquet(VAL_FILE)
    except FileNotFoundError as e:
        print(f"Error: Data file not found. Ensure '{TRAIN_FILE}' and '{VAL_FILE}' exist.")
        return

    # 2. Introduce a synthetic sensitive attribute
    np.random.seed(RANDOM_STATE)
    train_df[SENSITIVE_COLUMN] = np.random.randint(0, 2, train_df.shape[0])
    val_df[SENSITIVE_COLUMN] = np.random.randint(0, 2, val_df.shape[0])
    print(f"Added synthetic sensitive attribute: '{SENSITIVE_COLUMN}'")

    # 3. Prepare data for training and evaluation
    features = [col for col in train_df.columns if col not in [TARGET_COLUMN, 'Time', SENSITIVE_COLUMN]]
    X_train = train_df[features]
    y_train = train_df[TARGET_COLUMN]
    X_val = val_df[features]
    y_val = val_df[TARGET_COLUMN]
    sensitive_features_val = val_df[SENSITIVE_COLUMN]

    with mlflow.start_run(run_name="Baseline_Fairness_SHAP_Analysis"):
        # 4. Train a baseline model
        print("Training DecisionTreeClassifier...")
        model = DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=10)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]

        # 5. Log standard performance metrics
        performance_metrics = {
            "accuracy": accuracy_score(y_val, y_pred),
            "f1_score": f1_score(y_val, y_pred),
            "roc_auc": roc_auc_score(y_val, y_pred_proba)
        }
        mlflow.log_metrics(performance_metrics)
        print("Logged standard performance metrics.")

        # 6. Fairness Analysis with Fairlearn
        print("Performing fairness analysis with Fairlearn...")
        fairness_metrics = {
            "demographic_parity_difference": demographic_parity_difference(y_val, y_pred, sensitive_features=sensitive_features_val),
            "equalized_odds_difference": equalized_odds_difference(y_val, y_pred, sensitive_features=sensitive_features_val)
        }
        mlflow.log_metrics(fairness_metrics)
        print("Logged fairness metrics.")

        # 7. Create and log fairness plots and HTML report
        print("Creating fairness visualization plots...")
        html_report_path = create_fairness_plots(
            y_val, y_pred, y_pred_proba, sensitive_features_val, fairness_metrics
        )
        mlflow.log_artifact(html_report_path, "fairness_plots")
        print("  -> Generated and logged fairness HTML report.")

        # 8. Explainability Analysis with SHAP
        print("Performing explainability analysis with SHAP...")
        os.makedirs("shap_plots", exist_ok=True)
        
        explainer = shap.TreeExplainer(model)
        shap_values_raw = explainer.shap_values(X_val)
        
        # a. Generate and log feature importance summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values_raw, X_val, plot_type="bar", show=False)
        plt.title("SHAP Feature Importance Summary")
        plt.tight_layout()
        feature_importance_path = "shap_plots/feature_importance_summary.png"
        plt.savefig(feature_importance_path)
        plt.close()
        mlflow.log_artifact(feature_importance_path, "shap_plots")
        print("  -> Generated and logged SHAP feature importance plot.")

        # b. Generate and log interactive force plot for multiple samples
        print("Generating interactive force plot...")
        # Reshape the SHAP values as per your example to fix DimensionError
        shap_values_reshaped = np.array(shap_values_raw).transpose(2, 0, 1)
        
        # Generate force plot for the positive class (1) using the first 100 validation samples
        force_plot_object = shap.force_plot(
            explainer.expected_value[1],
            shap_values_reshaped[1][:100],  # Use reshaped values for class 1
            X_val.iloc[:100],
            show=False
        )
        force_plot_path = "shap_plots/interactive_force_plot.html"
        shap.save_html(force_plot_path, force_plot_object)
        mlflow.log_artifact(force_plot_path, "shap_plots")
        print("  -> Generated and logged interactive force plot.")

        # 9. Log the model
        mlflow.sklearn.log_model(model, "model")
        print("Logged model to MLflow.")

    print("\n✅ Fairness and Explainability experiment complete.")
    print("📊 Check MLflow UI for detailed results and artifacts!")

if __name__ == "__main__":
    run_fairness_experiment()

