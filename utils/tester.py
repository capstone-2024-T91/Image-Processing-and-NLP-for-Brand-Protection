import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import wandb
import sys
import os
from tqdm import tqdm 
import math
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.data_loader import load_test_data

class Tester:
    def __init__(self, model, model_name, test_files=['data/processed/phishing_test_1.csv','data/processed/phishing_test_2.csv'], project_name='capstone-2024', verbose=False, test_percentage=0.2, batch_size=15):
        """
        Initializes the Tester class.

        Args:
            model: The model to test.
            test_files: List of CSV files containing test data.
            model_name: Name of the model (for logging purposes).
            project_name: W&B project name.
            verbose: If True, prints detailed logs.
        """
        self.model = model
        self.test_files = test_files
        self.model_name = model_name
        self.project_name = project_name
        self.verbose = verbose
        self.test_percentage = test_percentage
        self.batch_size = batch_size

    def load_data(self):
        """
        Loads the test data from the CSV files.

        Returns:
            X_test: List of test data.
            y_test: List of test labels.
        """
        X_test, y_test = load_test_data(self.test_files, self.verbose, self.test_percentage)
        return X_test, y_test

    def evaluate(self):
        """
        Evaluates the model on the test data in batches and logs metrics to W&B after each batch.
        """
        X_test, y_test = self.load_data()
        total_samples = len(X_test)
        batch_size = self.batch_size
        num_batches = math.ceil(total_samples / batch_size)
        y_pred = []
        y_true = []
        if self.verbose:
            print("Starting predictions...")
            pbar = tqdm(total=total_samples, desc="Evaluating", unit="sample")
        else:
            pbar = None
        # Initialize W&B run
        wandb.init(project=self.project_name, name=self.model_name, job_type='eval')
        # Log test_percentage and batch_size as config parameters
        wandb.config.update({
            'test_percentage': self.test_percentage,
            'batch_size': self.batch_size
        })
        # Process data in batches
        for batch_num in range(num_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, total_samples)
            X_batch = X_test[start_idx:end_idx]
            y_batch = y_test[start_idx:end_idx]
            batch_pred = []
            for text in X_batch:
                processed_text = text
                prediction = self.model.predict(processed_text)
                batch_pred.append(int(prediction))
                if pbar:
                    pbar.update(1)
            # Update cumulative predictions and true labels
            y_pred.extend(batch_pred)
            y_true.extend(y_batch)
            # Compute cumulative metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            # Log metrics to W&B
            wandb.log({
                'batch': batch_num + 1,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })
            if self.verbose:
                print(f"Batch {batch_num + 1}/{num_batches} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
        # Finish W&B run
        wandb.finish()
        if pbar:
            pbar.close()
