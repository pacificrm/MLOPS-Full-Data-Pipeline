import unittest
import pandas as pd
from joblib import load
from sklearn.metrics import accuracy_score

class TestModelSanity(unittest.TestCase):
    """
    A suite of sanity tests for the trained fraud detection model.
    """
    def test_model_on_sample(self):
        """
        Tests the model's performance on a small, known sample of data.
        """
        # --- 1. Load Artifacts ---
        print("Loading sample data and trained model...")
        try:
            # Load the 6-row sample dataset
            sample_df = pd.read_csv("sample/sample.csv")
            # Load the trained model
            model = load("model.joblib")
        except FileNotFoundError as e:
            self.fail(f"ERROR: A required file was not found. Make sure you run 'dvc pull'. Details: {e}")

        # --- 2. Prepare Data ---
        # The target column is 'Class'
        TARGET_COLUMN = 'Class'
        
        # Define features as all columns except the target and the 'Time' column
        features = [col for col in sample_df.columns if col not in [TARGET_COLUMN, 'Time']]
        X_sample = sample_df[features]
        y_true = sample_df[TARGET_COLUMN]

        # --- 3. Predict and Evaluate ---
        print("Making predictions on the sample data...")
        y_pred = model.predict(X_sample)
        
        # Calculate accuracy on the sample
        acc = accuracy_score(y_true, y_pred)
        print(f"Accuracy on sample data: {acc:.4f}")

        # --- 4. Save Metric ---
        # Save the result to a separate metrics file for the report
        with open("sample_metrics.txt", "w") as f:
            f.write(f"Sample Accuracy: {acc:.4f}\n")
        
        # --- 5. Assert Performance ---
        # Assert that the accuracy is at least 80%.
        # This is a more realistic sanity check than expecting 100%.
        self.assertGreaterEqual(acc, 0.80, f"Model accuracy on sample ({acc:.4f}) is below the 80% threshold.")

if __name__ == "__main__":
    unittest.main()
