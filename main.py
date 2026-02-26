from dotenv import load_dotenv
import os
import json
from fastapi import FastAPI
import numpy as np 
import pandas as pd
from transaction_classifier import TransactionClassifier
from transaction import Transaction
from dataset_loader import DatasetLoader
from dataset_generator import DatasetGenerator
from dataset_loader import DatasetLoader 
from train_pipeline import TrainingPipeline
from datetime import datetime

load_dotenv()
app = FastAPI()


# Example: Complete training pipeline usage
def main():
    """Complete training pipeline demonstration."""
    
    # Initialize pipeline
    print("Initializing Training Pipeline...")
    pipeline = TrainingPipeline(
        model_dir="models",
        data_dir="data"
    )
    
    # Get model info
    model_info = pipeline.get_model_info()
    print(f"\nModel Info:")
    print(f"  - Trained: {model_info['is_trained']}")
    print(f"  - Last trained: {model_info['last_trained']}")
    print(f"  - Categories: {model_info['num_categories']}")
    print(f"  - Feedback entries: {model_info['feedback_entries']}")
    
    # Option 1: Run full training if no model exists
    if not model_info['is_trained']:
        print("\nNo trained model found. Running full training...")
        metrics = pipeline.run_full_training()
        print(f"Training completed with accuracy: {metrics.get('accuracy', 0):.3f}")
    
    # Option 2: Evaluate existing model
    print("\nEvaluating model performance...")
    eval_results = pipeline.evaluate_model()
    if eval_results:
        print(f"Model accuracy: {eval_results['accuracy']:.3f}")
    
    # Simulate user feedback
    print("\nSimulating user feedback...")
    sample_feedback = [
        {
            "transaction_id": "txn_001",
            "description": "Starbucks coffee purchase",
            "predicted_category": "entertainment",  # Wrong prediction
            "user_category": "dining",  # Correct category
            "was_correct": False,
            "user_id": "user_001"
        },
        {
            "transaction_id": "txn_002",
            "description": "Netflix monthly subscription",
            "predicted_category": "subscriptions",  # Correct prediction
            "user_category": "subscriptions",
            "was_correct": True,
            "user_id": "user_001"
        }
    ]
    
    # Log feedback
    pipeline.batch_log_feedback(sample_feedback)
    
    # Check if retraining should be triggered
    if pipeline._should_trigger_retraining():
        print("\nTriggering incremental retraining...")
        pipeline.run_incremental_training()
    
    # Trigger scheduled retraining
    print("\nChecking for scheduled retraining...")
    pipeline.trigger_scheduled_retraining()
    
    # Final model info
    final_info = pipeline.get_model_info()
    print(f"\nFinal Model State:")
    print(f"  - Last trained: {final_info['last_training_run'].get('timestamp', 'unknown')}")
    print(f"  - Training runs: {final_info['last_training_run'].get('run_type', 'unknown')}")
    print(f"  - Feedback processed: {final_info['feedback_entries']}")


@app.post('/categorize')
async def categorize_transaction(transaction: Transaction):
    """Categorize transaction using ML model"""
    categorizer = TransactionClassifier()
    category = categorizer.predict(transaction.description)
    confidence = categorizer.get_confidence()
    
    # Auto-learning from user corrections
    if transaction.user_category:
        categorizer.retrain(
            transaction.description,
            transaction.user_category
        )
    
    return {
        "predicted_category": category,
        "confidence": confidence,
        "suggested_budget": get_budget_suggestion(category)
    }



if __name__ == "__main__":
    main()